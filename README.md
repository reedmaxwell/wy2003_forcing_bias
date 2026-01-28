# WY2003 All-SNOTEL Forcing Bias Analysis

Compares CW3E atmospheric forcing against SNOTEL observations for **all SNOTEL sites** for Water Year 2003.

## Purpose

This is a standalone forcing bias characterization project, separate from the snow model sensitivity study. It provides a network-wide assessment of CW3E forcing quality.

## Key Results

**643 sites analyzed** (of 814 total)

| Metric | Daily | Annual/Seasonal |
|--------|-------|-----------------|
| Temperature Bias | +0.07°C | -0.21°C |
| Temperature Correlation | r = 0.97 | r = 0.93 |
| Precipitation Bias | -23% | -23% |
| Precipitation Correlation | r = 0.55 (log) | r = 0.93 |

### Bias Categories
| Category | Count | Percentage |
|----------|-------|------------|
| Low Bias | 111 | 17.3% |
| Temp Bias Only | 22 | 3.4% |
| Precip Bias Only | 424 | 65.9% |
| Both Biases | 86 | 13.4% |

### Geographic Patterns
- **Best precip (maritime):** WA, OR, CA - bias < 15%
- **Worst precip (continental):** MT, WY, NM - bias > 50%

## Setup

### 1. Install Dependencies

```bash
# Using conda (recommended)
conda create -n subsettools python=3.11
conda activate subsettools
pip install -r requirements.txt
conda install -c conda-forge cartopy  # For maps
```

### 2. Configure HydroData Credentials

Register at [https://hydrogen.princeton.edu/](https://hydrogen.princeton.edu/) to get API credentials.

Set environment variables:
```bash
export HF_EMAIL="your_email@example.com"
export HF_PIN="your_pin"
```

Or edit `config.py` directly (don't commit credentials to git).

### 3. Fetch Data

The forcing and observation data files are not included in the repo (~700 MB). Fetch them:

```bash
cd scripts

# Fetch site metadata (quick)
python fetch_snotel_metadata.py

# Fetch CW3E forcing (takes ~30 min for 800+ sites)
python fetch_forcing_wy2003.py

# Fetch SNOTEL observations
python fetch_snotel_obs_wy2003.py
```

### 4. Run Analysis

```bash
# Compute bias metrics
python analyze_forcing_bias.py

# Generate scatter plots, maps, and category lists
python create_scatter_map_categories.py
```

## Directory Structure

```
wy2003_forcing_bias/
├── config.py                           # Configuration
├── README.md
├── CLAUDE.md                           # AI assistant context
├── SESSION_NOTES.md                    # Development notes
├── requirements.txt
├── scripts/
│   ├── fetch_snotel_metadata.py        # Get all SNOTEL sites
│   ├── fetch_forcing_wy2003.py         # Fetch CW3E forcing
│   ├── fetch_snotel_obs_wy2003.py      # Fetch SNOTEL observations
│   ├── analyze_forcing_bias.py         # Compute bias metrics
│   └── create_scatter_map_categories.py # Generate plots and site lists
├── data/
│   ├── snotel_metadata.csv             # All SNOTEL sites (in git)
│   ├── forcing/                        # CW3E forcing files (NOT in git)
│   └── observations/                   # SNOTEL obs files (NOT in git)
└── results/
    ├── forcing_bias_wy2003.csv         # Full results (per-site bias)
    ├── forcing_bias_wy2003_summary.md  # Markdown report
    ├── sites_low_bias.csv              # 111 low-bias sites
    ├── sites_temp_bias_only.csv        # 22 temp-bias-only sites
    ├── sites_precip_bias_only.csv      # 424 precip-bias-only sites
    ├── sites_both_biases.csv           # 86 both-biases sites
    └── figures/
        ├── forcing_bias_wy2003_summary.png  # 4-panel summary
        ├── scatter_1to1_temp_daily.png      # Daily temp comparison
        ├── scatter_1to1_precip_daily.png    # Daily precip comparison
        ├── scatter_1to1_temp_seasonal.png   # Winter/Spring mean temp by site
        ├── scatter_1to1_precip_annual.png   # Annual precip totals by site
        └── site_map_by_bias.png             # Cartopy map with state boundaries
```

## Data Sources

All data fetched via HydroData (`hf_hydrodata`):

| Data | Dataset | Variables |
|------|---------|-----------|
| SNOTEL sites | `snotel` | Site metadata via `get_point_metadata()` |
| SNOTEL observations | `snotel` | Temperature, precipitation via `get_point_data()` |
| CW3E forcing | `CW3E` | 8 atmospheric variables via `get_gridded_data()` |

**Important:** HydroData returns precipitation in mm (no unit conversion needed).

## Bias Categories

Sites are categorized based on:
- **Temperature threshold:** |bias| < 1.0°C
- **Precipitation threshold:** |bias| < 10%

| Category | Description |
|----------|-------------|
| LOW_BIAS | Good for model evaluation |
| TEMP_BIAS_ONLY | Temperature biased, precip OK |
| PRECIP_BIAS_ONLY | Precip biased, temperature OK |
| BOTH_BIASES | Both significantly biased |

## Dependencies

- Python 3.11+
- pandas, numpy, matplotlib
- hf_hydrodata
- cartopy (for maps with Natural Earth boundaries)

## License

Internal use only.
