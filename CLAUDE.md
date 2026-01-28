# CLAUDE.md - Project Context for Claude Code

## Project Overview

This project compares CW3E atmospheric forcing data against SNOTEL observations for all SNOTEL sites (~814) for Water Year 2003. The goal is to characterize forcing bias across the entire network to identify sites suitable for snow model physics evaluation.

## Key Commands

```bash
cd ~/Projects/wy2003_forcing_bias/scripts
source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools

# Run bias analysis (requires data to be fetched first)
python analyze_forcing_bias.py

# Generate scatter plots, map, and category CSVs
python create_scatter_map_categories.py

# Fetch data (if not already present - large download)
python fetch_snotel_metadata.py
python fetch_forcing_wy2003.py      # Takes ~30 min for 800+ sites
python fetch_snotel_obs_wy2003.py
```

## Project Structure

```
wy2003_forcing_bias/
├── config.py          # Configuration (uses HF_EMAIL/HF_PIN env vars)
├── scripts/           # All Python scripts
├── data/              # Raw data (NOT in git - ~700MB, must be fetched)
│   ├── snotel_metadata.csv  # Site list (in git)
│   ├── forcing/       # CW3E forcing files (not in git)
│   └── observations/  # SNOTEL obs files (not in git)
└── results/           # Output CSVs and figures (in git)
```

## Key Files

| File | Purpose |
|------|---------|
| `analyze_forcing_bias.py` | Main analysis - computes per-site bias metrics |
| `create_scatter_map_categories.py` | Generates all plots and category CSVs |
| `results/forcing_bias_wy2003.csv` | Per-site bias results (643 sites) |
| `results/sites_low_bias.csv` | 111 sites good for model evaluation |
| `results/figures/site_map_by_bias.png` | Cartopy map showing geographic patterns |

## Important Technical Notes

1. **HydroData returns precipitation in mm** - DO NOT add unit conversion (bug was fixed Jan 2026)
2. **Cartopy required for maps** - `conda install -c conda-forge cartopy`
3. **Batch API requests** - HydroData needs batches of ~50 sites to avoid 414 URI Too Long errors
4. **Credentials** - Uses `HF_EMAIL` and `HF_PIN` environment variables (set in ~/.zshrc)

## Known Issues / Bug History

- **Precip unit conversion bug (fixed)**: Code was incorrectly multiplying some low-precip sites by 25.4, assuming inches→mm conversion. HydroData already returns mm. This caused annual precip correlation to drop from r=0.93 to r=0.08. Fixed by removing conversion logic.

## Current Results (WY2003)

| Metric | Value |
|--------|-------|
| Sites analyzed | 643 |
| Temperature bias | +0.07°C (excellent) |
| Precipitation bias | -25.9% (underestimate) |
| Low-bias sites | 111 (17.3%) |
| Daily T correlation | r = 0.97 |
| Annual P correlation | r = 0.93 |

### Geographic Patterns
- **Best precip:** WA (-5%), OR (-12%), CA (-11%) - maritime
- **Worst precip:** NM (-52%), MT (-51%), WY (-50%) - continental

## Bias Categories

Sites categorized by thresholds: |T bias| < 1.0°C, |P bias| < 10%

| Category | Count | Use Case |
|----------|-------|----------|
| LOW_BIAS | 111 | Best for model physics evaluation |
| TEMP_BIAS_ONLY | 22 | OK for precip-focused studies |
| PRECIP_BIAS_ONLY | 424 | OK for temp-focused studies |
| BOTH_BIASES | 86 | Use with caution |

## Code Style

- Python 3.11+
- pandas/numpy for data handling
- matplotlib for plotting
- cartopy for maps (Natural Earth boundaries)
- Scripts use `sys.path.insert(0, str(Path(__file__).parent.parent))` for config imports
