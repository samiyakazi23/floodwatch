#!/usr/bin/env bash
# Hand-labelled Sen1Floods11 chips (~700 MB) from the public Google Cloud bucket.
set -euo pipefail
mkdir -p data/sen1floods11 && cd data/sen1floods11
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand \
               gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand \
               gs://sen1floods11/v1.1/splits .
