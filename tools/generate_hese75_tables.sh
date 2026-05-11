#!/usr/bin/env bash

# Build HESE 7.5 derived detector-response tables from the public MC release.
# Usage:
#   bash tools/generate_hese75_tables.sh
# Optional:
#   HESE75_DATA_DIR=/path/to/.../resources/data bash tools/generate_hese75_tables.sh
#   OUTDIR=data/icecube/hese75/derived bash tools/generate_hese75_tables.sh

HESE75_DATA_DIR="${HESE75_DATA_DIR:-data/icecube/hese75/HESE-7-year-data-release-main/HESE-7-year-data-release/resources/data}"
OUTDIR="${OUTDIR:-data/icecube/hese75/derived}"

python tools/build_hese75_effective_area.py \
  --hese75-data-dir "${HESE75_DATA_DIR}" \
  --outdir "${OUTDIR}"

python tools/build_hese75_energy_resolution.py \
  --hese75-data-dir "${HESE75_DATA_DIR}" \
  --outdir "${OUTDIR}" \
  --morphology all

