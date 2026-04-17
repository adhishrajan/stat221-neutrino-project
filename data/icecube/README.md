HESE 7.5 Detector Response (Only)
=================================

This folder now keeps only the HESE 7.5-year release products used by the project.

Public data release:
- https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/

Derived files used by `config.yaml`:
- `data/icecube/hese75/derived/hese75_effective_area_trueE_allsky.txt`
- `data/icecube/hese75/derived/hese75_migration_all.txt`

Regenerate derived tables from the release MC:
- `bash tools/generate_hese75_tables.sh`

Or run builders directly:
- `python tools/build_hese75_effective_area.py`
- `python tools/build_hese75_energy_resolution.py --morphology all`
