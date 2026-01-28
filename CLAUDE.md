# CLAUDE.md - Project Context for Claude Code

## Project Overview

This project compares CW3E atmospheric forcing data against SNOTEL observations for all SNOTEL sites (~814) for Water Year 2003. The goal is to characterize forcing bias across the entire network.

## Key Commands

```bash
cd ~/Projects/wy2003_forcing_bias/scripts
source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools

# Run bias analysis (requires data to be fetched first)
python analyze_forcing_bias.py

# Generate scatter plots, map, and category CSVs
python create_scatter_map_categories.py

# Fetch data (if not already present)
python fetch_snotel_metadata.py
python fetch_forcing_wy2003.py      # Takes ~30 min for 800+ sites
python fetch_snotel_obs_wy2003.py
```

## Project Structure

- `config.py` - Configuration (paths, thresholds, HydroData credentials)
- `scripts/` - All Python scripts for fetching data and analysis
- `data/` - Raw data (not in git, must be fetched)
- `results/` - Output CSVs and figures (in git)

## Key Files

| File | Purpose |
|------|---------|
| `analyze_forcing_bias.py` | Main analysis - computes per-site bias metrics |
| `create_scatter_map_categories.py` | Generates all plots and category CSVs |
| `results/forcing_bias_wy2003.csv` | Per-site bias results |
| `results/sites_low_bias.csv` | 111 sites good for model evaluation |

## Important Technical Notes

1. **HydroData returns precipitation in mm** - no unit conversion needed
2. **Cartopy required for maps** - `conda install -c conda-forge cartopy`
3. **Batch API requests** - HydroData needs batches of ~50 sites to avoid 414 errors
4. **Credentials** - Set `HF_EMAIL` and `HF_PIN` environment variables (or edit config.py)

## Current Results (WY2003)

- 643 sites analyzed
- Temperature bias: +0.07°C (excellent)
- Precipitation bias: -25.9% (systematic underestimate)
- 111 low-bias sites identified
- Maritime sites (WA, OR, CA) have best precip agreement
- Continental sites (MT, WY, NM) have worst precip bias (~50%)

## Code Style

- Python 3.11+
- pandas/numpy for data handling
- matplotlib for plotting
- cartopy for maps (Natural Earth boundaries)
- Scripts use relative imports via sys.path manipulation
