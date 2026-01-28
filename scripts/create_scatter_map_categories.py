#!/usr/bin/env python3
"""
Create 1:1 scatter plots, spatial map, and categorized site lists.

Outputs:
- figures/scatter_1to1_temp_daily.png - Daily temperature: SNOTEL vs CW3E
- figures/scatter_1to1_precip_daily.png - Daily precipitation: SNOTEL vs CW3E
- figures/scatter_1to1_temp_seasonal.png - Winter/Spring mean temp per site
- figures/scatter_1to1_precip_annual.png - Annual precip totals per site
- figures/site_map_by_bias.png - Spatial map colored by bias category
- sites_low_bias.csv
- sites_temp_bias_only.csv
- sites_precip_bias_only.csv
- sites_both_biases.csv

Usage:
    cd ~/Projects/wy2003_forcing_bias/scripts
    source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools
    python create_scatter_map_categories.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config import (
    WATER_YEAR, START_DATE, END_DATE,
    TEMP_BIAS_THRESHOLD, PRECIP_BIAS_THRESHOLD,
    FORCING_DIR, OBS_DIR, RESULTS_DIR, FIGURES_DIR,
    SNOTEL_METADATA_FILE, FORCING_BIAS_RESULTS,
)


def load_forcing(site_id: str) -> pd.DataFrame:
    """Load CW3E forcing for a site and aggregate to daily."""
    safe_name = str(site_id).replace(" ", "_").replace("/", "_")
    filepath = FORCING_DIR / f"forcing_wy2003_{safe_name}.txt"

    if not filepath.exists():
        return None

    try:
        data = np.loadtxt(filepath)
        start = datetime(WATER_YEAR - 1, 10, 1, 0, 0)
        n_hours = len(data)
        dates = [start + timedelta(hours=i) for i in range(n_hours)]

        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates), columns=[
            'sw_down', 'lw_down', 'precip_mm_s', 'temp_k',
            'u_wind', 'v_wind', 'pressure', 'humidity'
        ])

        df['temp_c'] = df['temp_k'] - 273.15
        df['precip_mm_hr'] = df['precip_mm_s'] * 3600

        daily = df.resample('D').agg({
            'temp_c': 'mean',
            'precip_mm_hr': 'sum',
        })
        daily.rename(columns={'precip_mm_hr': 'precip_mm'}, inplace=True)
        return daily

    except Exception as e:
        return None


def load_snotel_obs():
    """Load SNOTEL temperature and precipitation observations."""
    temp_file = OBS_DIR / "snotel_temp_wy2003.csv"
    prec_file = OBS_DIR / "snotel_prec_wy2003.csv"

    temp_df = None
    prec_df = None

    if temp_file.exists():
        temp_df = pd.read_csv(temp_file)
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df.set_index('date', inplace=True)

    if prec_file.exists():
        prec_df = pd.read_csv(prec_file)
        prec_df['date'] = pd.to_datetime(prec_df['date'])
        prec_df.set_index('date', inplace=True)

    return temp_df, prec_df


def collect_daily_data(bias_df, snotel_temp, snotel_prec):
    """Collect all daily paired data for scatter plots."""
    all_temp_obs = []
    all_temp_cw3e = []
    all_prec_obs = []
    all_prec_cw3e = []

    for _, row in bias_df.iterrows():
        site_id = row['site_id']
        cw3e = load_forcing(site_id)
        if cw3e is None:
            continue

        # Temperature
        if snotel_temp is not None and site_id in snotel_temp.columns:
            obs_temp = snotel_temp[site_id].dropna()
            obs_temp = obs_temp[(obs_temp > -60) & (obs_temp < 60)]

            if len(obs_temp) > 0:
                # Convert F to C if needed
                if obs_temp.mean() > 50:
                    obs_temp = (obs_temp - 32) * 5/9

                common_dates = cw3e.index.intersection(obs_temp.index)
                if len(common_dates) > 0:
                    all_temp_obs.extend(obs_temp.loc[common_dates].values)
                    all_temp_cw3e.extend(cw3e.loc[common_dates, 'temp_c'].values)

        # Precipitation (HydroData returns mm - no conversion needed)
        if snotel_prec is not None and site_id in snotel_prec.columns:
            obs_prec = snotel_prec[site_id].dropna()
            obs_prec = obs_prec[(obs_prec >= 0) & (obs_prec < 500)]

            if len(obs_prec) > 0:
                common_dates = cw3e.index.intersection(obs_prec.index)
                if len(common_dates) > 0:
                    all_prec_obs.extend(obs_prec.loc[common_dates].values)
                    all_prec_cw3e.extend(cw3e.loc[common_dates, 'precip_mm'].values)

    return (np.array(all_temp_obs), np.array(all_temp_cw3e),
            np.array(all_prec_obs), np.array(all_prec_cw3e))


def collect_site_aggregates(bias_df, snotel_temp, snotel_prec):
    """Collect site-level aggregates: annual precip totals and winter/spring mean temp."""
    # Winter/Spring = Dec-May (months 12, 1, 2, 3, 4, 5)
    winter_spring_months = [12, 1, 2, 3, 4, 5]

    site_ids = []
    annual_prec_obs = []
    annual_prec_cw3e = []
    seasonal_temp_obs = []
    seasonal_temp_cw3e = []
    states = []

    for _, row in bias_df.iterrows():
        site_id = row['site_id']
        state = row.get('state', '')
        cw3e = load_forcing(site_id)
        if cw3e is None:
            continue

        # Annual precipitation totals (HydroData returns mm - no conversion needed)
        if snotel_prec is not None and site_id in snotel_prec.columns:
            obs_prec = snotel_prec[site_id].dropna()
            obs_prec = obs_prec[(obs_prec >= 0) & (obs_prec < 500)]

            if len(obs_prec) > 0:
                common_dates = cw3e.index.intersection(obs_prec.index)
                if len(common_dates) > 300:  # Need most of the year
                    obs_total = obs_prec.loc[common_dates].sum()
                    cw3e_total = cw3e.loc[common_dates, 'precip_mm'].sum()

                    site_ids.append(site_id)
                    states.append(state)
                    annual_prec_obs.append(obs_total)
                    annual_prec_cw3e.append(cw3e_total)

                    # Also get seasonal temp for this site
                    if snotel_temp is not None and site_id in snotel_temp.columns:
                        obs_temp = snotel_temp[site_id].dropna()
                        obs_temp = obs_temp[(obs_temp > -60) & (obs_temp < 60)]

                        if len(obs_temp) > 0:
                            if obs_temp.mean() > 50:
                                obs_temp = (obs_temp - 32) * 5/9

                            # Filter to winter/spring months
                            ws_dates = [d for d in cw3e.index.intersection(obs_temp.index)
                                       if d.month in winter_spring_months]

                            if len(ws_dates) > 100:  # Need reasonable coverage
                                obs_ws_mean = obs_temp.loc[ws_dates].mean()
                                cw3e_ws_mean = cw3e.loc[ws_dates, 'temp_c'].mean()
                                seasonal_temp_obs.append(obs_ws_mean)
                                seasonal_temp_cw3e.append(cw3e_ws_mean)
                            else:
                                seasonal_temp_obs.append(np.nan)
                                seasonal_temp_cw3e.append(np.nan)
                        else:
                            seasonal_temp_obs.append(np.nan)
                            seasonal_temp_cw3e.append(np.nan)
                    else:
                        seasonal_temp_obs.append(np.nan)
                        seasonal_temp_cw3e.append(np.nan)

    return {
        'site_ids': site_ids,
        'states': np.array(states),
        'annual_prec_obs': np.array(annual_prec_obs),
        'annual_prec_cw3e': np.array(annual_prec_cw3e),
        'seasonal_temp_obs': np.array(seasonal_temp_obs),
        'seasonal_temp_cw3e': np.array(seasonal_temp_cw3e),
    }


def create_scatter_temp_daily(temp_obs, temp_cw3e):
    """Create 1:1 scatter plot for daily temperature."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter with density coloring - use log scale and darker colormap
    hb = ax.hexbin(temp_obs, temp_cw3e, gridsize=80, cmap='viridis',
                   mincnt=1, norm=LogNorm())
    cb = plt.colorbar(hb, ax=ax, label='Count (log scale)')

    # 1:1 line
    lims = [min(temp_obs.min(), temp_cw3e.min()), max(temp_obs.max(), temp_cw3e.max())]
    ax.plot(lims, lims, 'r-', linewidth=2.5, label='1:1 line')

    # Linear fit
    mask = np.isfinite(temp_obs) & np.isfinite(temp_cw3e)
    z = np.polyfit(temp_obs[mask], temp_cw3e[mask], 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'w--', linewidth=2, label=f'Fit: y={z[0]:.2f}x + {z[1]:.2f}')

    # Stats
    bias = np.mean(temp_cw3e - temp_obs)
    rmse = np.sqrt(np.mean((temp_cw3e - temp_obs)**2))
    corr = np.corrcoef(temp_obs[mask], temp_cw3e[mask])[0, 1]

    stats_text = f'N = {len(temp_obs):,}\nBias = {bias:+.2f}°C\nRMSE = {rmse:.2f}°C\nr = {corr:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('SNOTEL Temperature (°C)', fontsize=12)
    ax.set_ylabel('CW3E Temperature (°C)', fontsize=12)
    ax.set_title(f'Daily Temperature: SNOTEL vs CW3E - WY{WATER_YEAR}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "scatter_1to1_temp_daily.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def create_scatter_precip_daily(prec_obs, prec_cw3e):
    """Create 1:1 scatter plot for daily precipitation."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(10, 10))

    # Filter to positive values for log scale
    mask = (prec_obs > 0.1) & (prec_cw3e > 0.1)
    prec_obs_pos = prec_obs[mask]
    prec_cw3e_pos = prec_cw3e[mask]

    # Scatter with density coloring - use log scale for color and darker cmap
    hb = ax.hexbin(prec_obs_pos, prec_cw3e_pos, gridsize=60, cmap='viridis',
                   mincnt=1, xscale='log', yscale='log', norm=LogNorm())
    cb = plt.colorbar(hb, ax=ax, label='Count (log scale)')

    # 1:1 line
    lims = [0.1, max(prec_obs_pos.max(), prec_cw3e_pos.max())]
    ax.plot(lims, lims, 'r-', linewidth=2.5, label='1:1 line')

    # Stats (on all data including zeros)
    total_obs = prec_obs.sum()
    total_cw3e = prec_cw3e.sum()
    bias_pct = (total_cw3e - total_obs) / total_obs * 100

    # Correlation on log-transformed positive values
    corr = np.corrcoef(np.log10(prec_obs_pos), np.log10(prec_cw3e_pos))[0, 1]

    stats_text = (f'N (>0.1mm) = {len(prec_obs_pos):,}\n'
                  f'Total bias = {bias_pct:+.1f}%\n'
                  f'r (log) = {corr:.3f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('SNOTEL Precipitation (mm/day)', fontsize=12)
    ax.set_ylabel('CW3E Precipitation (mm/day)', fontsize=12)
    ax.set_title(f'Daily Precipitation: SNOTEL vs CW3E - WY{WATER_YEAR}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    outpath = FIGURES_DIR / "scatter_1to1_precip_daily.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def create_scatter_precip_annual(site_data):
    """Create 1:1 scatter plot for annual precipitation totals per site."""
    prec_obs = site_data['annual_prec_obs']
    prec_cw3e = site_data['annual_prec_cw3e']
    states = site_data['states']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by region
    region_colors = {
        'PNW': '#1f77b4',    # WA, OR - blue
        'CA_NV': '#ff7f0e',  # CA, NV - orange
        'ROCKIES': '#2ca02c', # MT, ID, WY, CO, UT - green
        'SW': '#d62728',     # AZ, NM - red
        'OTHER': '#7f7f7f',  # SD, etc - gray
    }

    def get_region(state):
        if state in ['WA', 'OR']:
            return 'PNW'
        elif state in ['CA', 'NV']:
            return 'CA_NV'
        elif state in ['MT', 'ID', 'WY', 'CO', 'UT']:
            return 'ROCKIES'
        elif state in ['AZ', 'NM']:
            return 'SW'
        else:
            return 'OTHER'

    regions = np.array([get_region(s) for s in states])

    for region, color in region_colors.items():
        mask = regions == region
        if mask.sum() > 0:
            region_names = {'PNW': 'Pacific NW (WA, OR)',
                          'CA_NV': 'CA & NV',
                          'ROCKIES': 'Rockies (MT, ID, WY, CO, UT)',
                          'SW': 'Southwest (AZ, NM)',
                          'OTHER': 'Other'}
            ax.scatter(prec_obs[mask], prec_cw3e[mask],
                      c=color, label=f"{region_names[region]} ({mask.sum()})",
                      s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 1:1 line
    max_val = max(prec_obs.max(), prec_cw3e.max())
    ax.plot([0, max_val], [0, max_val], 'r-', linewidth=2, label='1:1 line')

    # Linear fit
    mask = np.isfinite(prec_obs) & np.isfinite(prec_cw3e)
    z = np.polyfit(prec_obs[mask], prec_cw3e[mask], 1)
    p = np.poly1d(z)
    ax.plot([0, max_val], p([0, max_val]), 'k--', linewidth=1.5,
            label=f'Fit: y={z[0]:.2f}x + {z[1]:.0f}')

    # Stats
    bias_mm = np.mean(prec_cw3e - prec_obs)
    bias_pct = (np.sum(prec_cw3e) - np.sum(prec_obs)) / np.sum(prec_obs) * 100
    rmse = np.sqrt(np.mean((prec_cw3e - prec_obs)**2))
    corr = np.corrcoef(prec_obs[mask], prec_cw3e[mask])[0, 1]

    stats_text = (f'N sites = {len(prec_obs)}\n'
                  f'Bias = {bias_mm:+.0f} mm ({bias_pct:+.1f}%)\n'
                  f'RMSE = {rmse:.0f} mm\n'
                  f'r = {corr:.3f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('SNOTEL Annual Precipitation (mm)', fontsize=12)
    ax.set_ylabel('CW3E Annual Precipitation (mm)', fontsize=12)
    ax.set_title(f'Annual Precipitation by Site: SNOTEL vs CW3E - WY{WATER_YEAR}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "scatter_1to1_precip_annual.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def create_scatter_temp_seasonal(site_data):
    """Create 1:1 scatter plot for winter/spring mean temperature per site."""
    temp_obs = site_data['seasonal_temp_obs']
    temp_cw3e = site_data['seasonal_temp_cw3e']
    states = site_data['states']

    # Filter to valid data
    valid = np.isfinite(temp_obs) & np.isfinite(temp_cw3e)
    temp_obs = temp_obs[valid]
    temp_cw3e = temp_cw3e[valid]
    states = states[valid]

    if len(temp_obs) == 0:
        print("No valid seasonal temperature data")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by region
    region_colors = {
        'PNW': '#1f77b4',
        'CA_NV': '#ff7f0e',
        'ROCKIES': '#2ca02c',
        'SW': '#d62728',
        'OTHER': '#7f7f7f',
    }

    def get_region(state):
        if state in ['WA', 'OR']:
            return 'PNW'
        elif state in ['CA', 'NV']:
            return 'CA_NV'
        elif state in ['MT', 'ID', 'WY', 'CO', 'UT']:
            return 'ROCKIES'
        elif state in ['AZ', 'NM']:
            return 'SW'
        else:
            return 'OTHER'

    regions = np.array([get_region(s) for s in states])

    for region, color in region_colors.items():
        mask = regions == region
        if mask.sum() > 0:
            region_names = {'PNW': 'Pacific NW (WA, OR)',
                          'CA_NV': 'CA & NV',
                          'ROCKIES': 'Rockies (MT, ID, WY, CO, UT)',
                          'SW': 'Southwest (AZ, NM)',
                          'OTHER': 'Other'}
            ax.scatter(temp_obs[mask], temp_cw3e[mask],
                      c=color, label=f"{region_names[region]} ({mask.sum()})",
                      s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 1:1 line
    min_val = min(temp_obs.min(), temp_cw3e.min())
    max_val = max(temp_obs.max(), temp_cw3e.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 line')

    # Linear fit
    z = np.polyfit(temp_obs, temp_cw3e, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], p([min_val, max_val]), 'k--', linewidth=1.5,
            label=f'Fit: y={z[0]:.2f}x + {z[1]:.2f}')

    # Stats
    bias = np.mean(temp_cw3e - temp_obs)
    rmse = np.sqrt(np.mean((temp_cw3e - temp_obs)**2))
    corr = np.corrcoef(temp_obs, temp_cw3e)[0, 1]

    stats_text = (f'N sites = {len(temp_obs)}\n'
                  f'Bias = {bias:+.2f}°C\n'
                  f'RMSE = {rmse:.2f}°C\n'
                  f'r = {corr:.3f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('SNOTEL Winter/Spring Mean Temperature (°C)', fontsize=12)
    ax.set_ylabel('CW3E Winter/Spring Mean Temperature (°C)', fontsize=12)
    ax.set_title(f'Winter/Spring (Dec-May) Mean Temperature by Site - WY{WATER_YEAR}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "scatter_1to1_temp_seasonal.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def create_spatial_map(bias_df):
    """Create spatial map of sites colored by bias category using cartopy."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Create figure with PlateCarree projection
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set map extent (western US)
    ax.set_extent([-125, -102, 31, 50], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)

    # Add state labels
    state_labels = {
        'WA': (-120.5, 47.5), 'OR': (-120.5, 44), 'CA': (-119.5, 37),
        'NV': (-117, 39.5), 'ID': (-114.5, 44.5), 'MT': (-110, 47),
        'WY': (-107.5, 43), 'UT': (-111.5, 39.5), 'CO': (-105.5, 39),
        'AZ': (-111.5, 34.5), 'NM': (-106, 34.5), 'SD': (-100.5, 44.5),
    }
    for state, (lon, lat) in state_labels.items():
        ax.text(lon, lat, state, fontsize=10, fontweight='bold', color='dimgray',
                ha='center', va='center', alpha=0.7, transform=ccrs.PlateCarree())

    category_colors = {
        'LOW_BIAS': '#2ca02c',          # Green
        'TEMP_BIAS_ONLY': '#ff7f0e',    # Orange
        'PRECIP_BIAS_ONLY': '#9467bd',  # Purple
        'BOTH_BIASES': '#d62728',       # Red
    }

    category_labels = {
        'LOW_BIAS': 'Low Bias',
        'TEMP_BIAS_ONLY': 'Temp Bias Only',
        'PRECIP_BIAS_ONLY': 'Precip Bias Only',
        'BOTH_BIASES': 'Both Biases',
    }

    # Plot each category (low bias on top)
    for cat in ['BOTH_BIASES', 'PRECIP_BIAS_ONLY', 'TEMP_BIAS_ONLY', 'LOW_BIAS']:
        mask = bias_df['bias_category'] == cat
        if mask.sum() > 0:
            subset = bias_df[mask]
            ax.scatter(subset['longitude'], subset['latitude'],
                      c=category_colors[cat],
                      label=f"{category_labels[cat]} ({mask.sum()})",
                      s=50, alpha=0.8, edgecolors='white', linewidth=0.4,
                      zorder=3 if cat == 'LOW_BIAS' else 2,
                      transform=ccrs.PlateCarree())

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 10}
    gl.ylabel_style = {'fontsize': 10}

    ax.set_title(f'SNOTEL Sites by Forcing Bias Category - WY{WATER_YEAR}\n'
                 f'(|T bias| < {TEMP_BIAS_THRESHOLD}°C, |P bias| < {PRECIP_BIAS_THRESHOLD}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()
    outpath = FIGURES_DIR / "site_map_by_bias.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def export_category_lists(bias_df):
    """Export CSV files for each bias category."""
    cols = ['site_id', 'site_name', 'state', 'latitude', 'longitude',
            'elevation_m', 'temp_bias_c', 'prec_total_bias_pct']

    categories = {
        'LOW_BIAS': 'sites_low_bias.csv',
        'TEMP_BIAS_ONLY': 'sites_temp_bias_only.csv',
        'PRECIP_BIAS_ONLY': 'sites_precip_bias_only.csv',
        'BOTH_BIASES': 'sites_both_biases.csv',
    }

    for cat, filename in categories.items():
        mask = bias_df['bias_category'] == cat
        subset = bias_df.loc[mask, cols].copy()
        subset = subset.sort_values(['state', 'site_name'])

        outpath = RESULTS_DIR / filename
        subset.to_csv(outpath, index=False)
        print(f"Saved: {outpath} ({len(subset)} sites)")


def main():
    print("=" * 70)
    print(f"Creating Scatter Plots, Map, and Category Lists - WY{WATER_YEAR}")
    print("=" * 70)

    # Load bias results
    if not FORCING_BIAS_RESULTS.exists():
        print(f"ERROR: Results not found: {FORCING_BIAS_RESULTS}")
        print("Run analyze_forcing_bias.py first")
        return

    bias_df = pd.read_csv(FORCING_BIAS_RESULTS)
    print(f"Loaded {len(bias_df)} sites from bias results")

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load SNOTEL observations
    print("\nLoading SNOTEL observations...")
    snotel_temp, snotel_prec = load_snotel_obs()

    # Collect daily paired data for scatter plots
    print("\nCollecting daily data for scatter plots...")
    temp_obs, temp_cw3e, prec_obs, prec_cw3e = collect_daily_data(
        bias_df, snotel_temp, snotel_prec
    )

    print(f"  Temperature pairs: {len(temp_obs):,}")
    print(f"  Precipitation pairs: {len(prec_obs):,}")

    # Collect site-level aggregates
    print("\nCollecting site-level aggregates...")
    site_data = collect_site_aggregates(bias_df, snotel_temp, snotel_prec)
    print(f"  Sites with annual precip: {len(site_data['annual_prec_obs'])}")
    n_temp = np.sum(np.isfinite(site_data['seasonal_temp_obs']))
    print(f"  Sites with seasonal temp: {n_temp}")

    # Create scatter plots
    print("\nCreating daily scatter plots...")
    if len(temp_obs) > 0:
        create_scatter_temp_daily(temp_obs, temp_cw3e)
    if len(prec_obs) > 0:
        create_scatter_precip_daily(prec_obs, prec_cw3e)

    print("\nCreating site-level scatter plots...")
    if len(site_data['annual_prec_obs']) > 0:
        create_scatter_precip_annual(site_data)
    if n_temp > 0:
        create_scatter_temp_seasonal(site_data)

    # Create spatial map
    print("\nCreating spatial map...")
    create_spatial_map(bias_df)

    # Export category lists
    print("\nExporting category lists...")
    export_category_lists(bias_df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    for cat in ['LOW_BIAS', 'TEMP_BIAS_ONLY', 'PRECIP_BIAS_ONLY', 'BOTH_BIASES']:
        count = (bias_df['bias_category'] == cat).sum()
        print(f"  {cat.replace('_', ' ').title():20s}: {count:4d} sites")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
