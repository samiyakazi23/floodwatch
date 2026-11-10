"""Flood mapping from Sentinel-1 radar (VV, VH) with a U-Net.

Radar sees through monsoon cloud, which is why agencies map floods with it.
Calm water reflects the radar pulse away from the satellite, so it shows up
dark; the hard cases are wet soil, shadows and urban areas.

Data: Sen1Floods11 hand-labelled chips (446 scenes, 11 flood events incl. Assam).
Baseline: Otsu thresholding on VH, the classic operational method.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

ROOT = Path("data/sen1floods11")
# per-band stats (dB) used for normalisation, computed on the train split
MEAN = np.array([-9.2, -15.8], dtype=np.float32)
STD = np.array([3.3, 3.7], dtype=np.float32)
IGNORE = -1


def read_split(name: str) -> list[str]:
    path = ROOT / "splits" / "flood_handlabeled" / f"flood_{name}_data.csv"
    return [row[0].replace("_S1Hand.tif", "") for row in csv.reader(open(path))]


def load_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(ROOT / "S1Hand" / f"{chip_id}_S1Hand.tif") as s:
        x = s.read().astype(np.float32)
    with rasterio.open(ROOT / "LabelHand" / f"{chip_id}_LabelHand.tif") as s:
        y = s.read(1).astype(np.int64)
    return x, y


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=-30.0)
    x = np.clip(x, -50.0, 5.0)
    return (x - MEAN[:, None, None]) / STD[:, None, None]


def build_model() -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    # compact encoder so it trains on a laptop CPU; no ImageNet weights (radar isn't RGB)
    return smp.Unet("resnet18", encoder_weights=None, in_channels=2, classes=2)


# ---------- metrics ----------

def confusion(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y != IGNORE
    p, t = pred[m].astype(int), y[m].astype(int)
    return np.bincount(t * 2 + p, minlength=4).reshape(2, 2)  # rows=truth, cols=pred


def scores(cm: np.ndarray) -> dict:
    tn, fp, fn, tp = cm.ravel()
    return {
        "water_iou": round(tp / max(tp + fp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "accuracy": round((tp + tn) / max(cm.sum(), 1), 4),
    }


def otsu_predict(x: np.ndarray) -> np.ndarray:
    """Classic baseline: per-chip Otsu threshold on VH backscatter."""
    vh = np.nan_to_num(x[1], nan=0.0)
    vals = vh[vh < 0]
    if vals.size < 100:
        return np.zeros_like(vh, dtype=np.int64)
    hist, edges = np.histogram(vals, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    thr = centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
    return (vh < thr).astype(np.int64)


# ---------- training ----------

class Chips(torch.utils.data.Dataset):
    def __init__(self, ids: list[str], crop: int | None = 256, augment: bool = True):
        self.data = [load_chip(i) for i in ids]
        self.ids, self.crop, self.augment = ids, crop, augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        x = normalise(x)
        if self.crop:
            # bias crops toward pixels that are labelled
            for _ in range(5):
                r, c = np.random.randint(0, 512 - self.crop, 2)
                if (y[r:r + self.crop, c:c + self.crop] != IGNORE).mean() > 0.3:
                    break
            x, y = x[:, r:r + self.crop, c:c + self.crop], y[r:r + self.crop, c:c + self.crop]
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[::-1]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = (y != IGNORE).float()
    p = logits.softmax(1)[:, 1] * m
    t = (y == 1).float() * m
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full 512x512 chip -> (water probability, mask)."""
    model.eval()
    t = torch.from_numpy(normalise(x))[None]
    prob = model(t).softmax(1)[0, 1].numpy()
    return prob, (prob > 0.5).astype(np.int64)


def evaluate(model, ids: list[str]) -> dict:
    cm_m, cm_o = np.zeros((2, 2), int), np.zeros((2, 2), int)
    per_event: dict[str, np.ndarray] = {}
    for cid in ids:
        x, y = load_chip(cid)
        _, pred = predict(model, x)
        c = confusion(pred, y)
        cm_m += c
        cm_o += confusion(otsu_predict(x), y)
        ev = cid.split("_")[0]
        per_event[ev] = per_event.get(ev, np.zeros((2, 2), int)) + c
    return {
        "unet": scores(cm_m),
        "otsu_baseline": scores(cm_o),
        "per_event_iou": {k: scores(v)["water_iou"] for k, v in sorted(per_event.items())},
    }


def train(epochs: int = 40, batch: int = 8, lr: float = 1e-3, out: Path = Path("models/flood_unet.pt"), threads: int = 2) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(0); np.random.seed(0)
    tr = Chips(read_split("train"))
    loader = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    val_ids = read_split("valid")
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    weight = torch.tensor([1.0, 2.0])  # water is the minority class
    best, history = -1.0, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); losses = []
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weight, ignore_index=IGNORE) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        row = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4), "secs": round(time.time() - t0, 1)}
        if (ep + 1) % 4 == 0 or ep + 1 == epochs:
            row["val_iou"] = evaluate(model, val_ids)["unet"]["water_iou"]
            if row["val_iou"] > best:
                best = row["val_iou"]
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out)
        history.append(row)
        print(json.dumps(row), flush=True)
    return {"best_val_iou": best, "history": history}


def load_model(path: Path = Path("models/flood_unet.pt")) -> torch.nn.Module:
    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    t = train(a.epochs)
    model = load_model()
    report = {
        "train": t,
        "test": evaluate(model, read_split("test")),
        "bolivia_unseen_event": evaluate(model, read_split("bolivia")),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/segmentation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "train"}, indent=2))
