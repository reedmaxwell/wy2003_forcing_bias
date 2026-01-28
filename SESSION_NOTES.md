# WY2003 Forcing Bias Analysis - Session Notes

**Last Updated:** January 27, 2026
**Status:** Analysis complete, all plots and exports generated

---

## Project Purpose

Standalone forcing bias analysis for **ALL SNOTEL sites** (~814) for **Water Year 2003 only**. This is separate from the `snow_model_sensitivity` project which focuses on 27 selected sites across multiple years.

Goal: Characterize CW3E forcing bias across the entire SNOTEL network for a single year.

---

## Session Log - January 27, 2026 (Evening)

### Work Completed

1. **Created 1:1 scatter plots**
   - `scatter_1to1_temp_daily.png` - Daily temperature (209,420 pairs)
   - `scatter_1to1_precip_daily.png` - Daily precipitation (232,474 pairs)
   - `scatter_1to1_temp_seasonal.png` - Winter/Spring (Dec-May) mean temp per site
   - `scatter_1to1_precip_annual.png` - Annual precipitation totals per site

2. **Created spatial map with cartopy**
   - `site_map_by_bias.png` - Uses Natural Earth boundaries (state lines, coastlines, lakes)
   - Shows clear maritime vs continental divide in forcing quality

3. **Exported site category lists**
   - `sites_low_bias.csv` (111 sites)
   - `sites_temp_bias_only.csv` (22 sites)
   - `sites_precip_bias_only.csv` (424 sites)
   - `sites_both_biases.csv` (86 sites)

4. **Fixed critical bug: precipitation unit conversion**
   - **Problem:** Code was incorrectly converting some sites from "inches to mm" when HydroData already returns mm
   - **Symptom:** 18 low-precip sites were being multiplied by 25.4, creating extreme outliers (e.g., 700 mm becoming 17,800 mm)
   - **Fix:** Removed erroneous conversion logic in `analyze_forcing_bias.py` and `create_scatter_map_categories.py`
   - **Result:** Precip bias corrected from -27.9% to -25.9%, correlation improved from r=0.08 to r=0.93

5. **Improved plot aesthetics**
   - Daily scatter plots now use viridis colormap with log scale for better visibility
   - Larger figure sizes and better labeling

### Key Finding: Daily vs Annual Comparison

| Metric | Daily | Annual (per site) |
|--------|-------|-------------------|
| Precip Bias | -23% | -23% |
| Precip Correlation | r = 0.55 (log) | r = 0.93 |

The bias is consistent, but the **daily correlation is much lower** due to timing offsets. Annual totals reveal the "true" systematic underestimate without timing noise.

---

## Current Results (Corrected)

**643 sites analyzed** (of 814 total)

| Metric | Value |
|--------|-------|
| Mean Temperature Bias | **+0.07°C** (excellent) |
| Mean Precipitation Bias | **-25.9%** (underestimate) |
| Low-bias sites | **111 (17.3%)** |

### Bias Categories
| Category | Count | Percentage |
|----------|-------|------------|
| Low Bias | 111 | 17.3% |
| Temp Bias Only | 22 | 3.4% |
| Precip Bias Only | 424 | 65.9% |
| Both Biases | 86 | 13.4% |

### Geographic Patterns
- **Best precip (maritime):** WA (-4.7%), OR (-11.5%), CA (-11.2%)
- **Worst precip (continental):** NM (-51.9%), MT (-51.3%), WY (-49.8%)

---

## Project Location

```
~/Projects/wy2003_forcing_bias/
├── config.py                    # Configuration (WY2003, thresholds, paths)
├── README.md                    # Project overview
├── SESSION_NOTES.md             # This file
├── scripts/
│   ├── fetch_snotel_metadata.py # Get all SNOTEL sites from HydroData
│   ├── fetch_forcing_wy2003.py  # Fetch CW3E forcing
│   ├── fetch_snotel_obs_wy2003.py # Fetch SNOTEL temp & precip
│   ├── analyze_forcing_bias.py  # Compare and compute metrics
│   ├── create_scatter_map_categories.py # Generate all plots and CSVs
│   ├── check_snotel_variables.py # Diagnostic script
│   └── compare_sensitivity_sites.py # Compare with snow_model_sensitivity sites
├── data/
│   ├── snotel_metadata.csv      # 814 SNOTEL sites
│   ├── forcing/                 # CW3E forcing files (814 files)
│   └── observations/            # SNOTEL obs (temp, precip CSVs)
└── results/
    ├── forcing_bias_wy2003.csv
    ├── forcing_bias_wy2003_summary.md
    ├── sites_low_bias.csv
    ├── sites_temp_bias_only.csv
    ├── sites_precip_bias_only.csv
    ├── sites_both_biases.csv
    └── figures/
        ├── forcing_bias_wy2003_summary.png  # 4-panel overview
        ├── scatter_1to1_temp_daily.png      # Daily T comparison
        ├── scatter_1to1_precip_daily.png    # Daily P comparison
        ├── scatter_1to1_temp_seasonal.png   # Seasonal T by site
        ├── scatter_1to1_precip_annual.png   # Annual P by site
        └── site_map_by_bias.png             # Cartopy map
```

---

## Data Sources (all via HydroData)

| Data | Method | Notes |
|------|--------|-------|
| Site metadata | `hf.get_point_metadata(dataset="snotel", variable="swe", ...)` | 814 sites |
| CW3E forcing | `hf.get_gridded_data(dataset="CW3E", ...)` | 8 variables, hourly |
| SNOTEL temp | `hf.get_point_data(variable="air_temp", aggregation="mean")` | Daily mean °C |
| SNOTEL precip | `hf.get_point_data(variable="precipitation", aggregation="sum")` | Daily total mm |

**Important HydroData notes:**
- SNOTEL variables require specific aggregations (not "sod")
- `air_temp`: use `aggregation="mean"` (or "maximum", "minimum")
- `precipitation`: use `aggregation="sum"` (NOT "accumulated" which gives cumulative since station start)
- **HydroData returns precipitation in mm** - no unit conversion needed!
- Must batch requests (~50 sites per batch) to avoid 414 URI Too Long errors

---

## Bugs Fixed

### Session 1 (Earlier)
1. **414 URI Too Long** - Fixed by batching site requests (50 per batch)
2. **Wrong aggregation for SNOTEL obs** - Temperature: `aggregation="mean"`, Precipitation: `aggregation="sum"`
3. **Temperature outliers (-551°C)** - Bad data flags filtered: `obs_temp[(obs_temp > -60) & (obs_temp < 60)]`
4. **Precip outliers** - Using "sum" not "accumulated" for daily totals
5. **Elevation empty** - Metadata column is `usda_elevation` (in feet), converted to meters

### Session 2 (Tonight)
6. **Erroneous unit conversion** - Removed incorrect inches→mm conversion (HydroData already returns mm)
   - Was causing 18 sites to have 25x inflated precip values
   - Annual scatter correlation jumped from r=0.08 to r=0.93 after fix

---

## Key Commands

```bash
cd ~/Projects/wy2003_forcing_bias/scripts
source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools

# Run full analysis
python analyze_forcing_bias.py

# Generate scatter plots, map, and category CSVs
python create_scatter_map_categories.py
```

---

## Key Findings

1. **Temperature forcing is excellent** - near-zero mean bias (+0.07°C), r=0.97 daily, symmetric distribution
2. **Precipitation forcing systematically low** - 26% underestimate on average
3. **Daily timing offsets add noise** - r=0.55 daily vs r=0.93 annual for precip
4. **Maritime sites (WA, OR, CA) have best precip** - closer to observations
5. **Continental sites (MT, WY, NM) have worst precip bias** - 50%+ underestimate
6. **No strong elevation dependence** for either temp or precip bias
7. **111 "gold standard" low-bias sites** available for model physics evaluation

---

## Next Steps / Ideas

- [x] Create spatial map of bias by site (done - with cartopy)
- [x] Create 1:1 scatter plots for T and P (done - daily and annual)
- [x] Export categorized site lists (done - 4 CSVs)
- [ ] Investigate why maritime sites have better precip
- [ ] Compare with other water years
- [ ] Use low-bias sites for model physics evaluation
