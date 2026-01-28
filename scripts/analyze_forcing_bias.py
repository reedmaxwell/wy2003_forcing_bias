#!/usr/bin/env python3
"""
Analyze CW3E forcing bias vs SNOTEL observations for WY2003.

Compares:
- Temperature: CW3E hourly (aggregated to daily) vs SNOTEL daily
- Precipitation: CW3E hourly (summed to daily) vs SNOTEL daily accumulation

Usage:
    cd ~/Projects/wy2003_forcing_bias/scripts
    source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools
    python analyze_forcing_bias.py
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
    SNOTEL_METADATA_FILE,
    FORCING_BIAS_RESULTS, FORCING_BIAS_SUMMARY,
)


def load_forcing(site_id: str) -> pd.DataFrame:
    """Load CW3E forcing for a site and aggregate to daily."""
    safe_name = str(site_id).replace(" ", "_").replace("/", "_")
    filepath = FORCING_DIR / f"forcing_wy2003_{safe_name}.txt"

    if not filepath.exists():
        return None

    try:
        data = np.loadtxt(filepath)

        # Create hourly datetime index
        start = datetime(WATER_YEAR - 1, 10, 1, 0, 0)
        n_hours = len(data)
        dates = [start + timedelta(hours=i) for i in range(n_hours)]

        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates), columns=[
            'sw_down', 'lw_down', 'precip_mm_s', 'temp_k',
            'u_wind', 'v_wind', 'pressure', 'humidity'
        ])

        df['temp_c'] = df['temp_k'] - 273.15
        df['precip_mm_hr'] = df['precip_mm_s'] * 3600

        # Aggregate to daily
        daily = df.resample('D').agg({
            'temp_c': 'mean',
            'precip_mm_hr': 'sum',
        })
        daily.rename(columns={'precip_mm_hr': 'precip_mm'}, inplace=True)

        return daily

    except Exception as e:
        print(f"  Error loading forcing for {site_id}: {e}")
        return None


def load_snotel_obs() -> tuple:
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


def analyze_site(site_id: str, site_info: dict, cw3e_daily: pd.DataFrame,
                 snotel_temp: pd.DataFrame, snotel_prec: pd.DataFrame) -> dict:
    """Analyze bias for a single site."""

    # Get elevation - metadata has 'usda_elevation' in feet
    elev_ft = site_info.get('usda_elevation', np.nan)
    elev_m = elev_ft * 0.3048 if not pd.isna(elev_ft) else np.nan

    results = {
        'site_id': site_id,
        'site_name': site_info.get('site_name', ''),
        'latitude': site_info.get('latitude', np.nan),
        'longitude': site_info.get('longitude', np.nan),
        'elevation_m': elev_m,
        'state': site_info.get('state', ''),
    }

    # Temperature comparison
    if snotel_temp is not None and site_id in snotel_temp.columns:
        obs_temp = snotel_temp[site_id].dropna()

        # Filter out bad values (missing data flags like -999)
        obs_temp = obs_temp[(obs_temp > -60) & (obs_temp < 60)]

        if len(obs_temp) > 0:
            # Convert if needed (SNOTEL often in °F)
            if obs_temp.mean() > 50:  # Likely Fahrenheit
                obs_temp = (obs_temp - 32) * 5/9

            common_dates = cw3e_daily.index.intersection(obs_temp.index)

            if len(common_dates) > 30:  # Need at least 30 days
                cw3e_temp = cw3e_daily.loc[common_dates, 'temp_c']
                snotel_t = obs_temp.loc[common_dates]

                results['temp_bias_c'] = (cw3e_temp - snotel_t).mean()
                results['temp_mae_c'] = (cw3e_temp - snotel_t).abs().mean()
                results['temp_rmse_c'] = np.sqrt(((cw3e_temp - snotel_t)**2).mean())
                results['temp_corr'] = cw3e_temp.corr(snotel_t)
                results['temp_n_days'] = len(common_dates)

    # Precipitation comparison
    if snotel_prec is not None and site_id in snotel_prec.columns:
        obs_prec = snotel_prec[site_id].dropna()

        # Filter out bad values (negative precip or unrealistic daily values)
        obs_prec = obs_prec[(obs_prec >= 0) & (obs_prec < 500)]  # < 500mm/day is reasonable max

        if len(obs_prec) > 0:
            # HydroData returns precipitation in mm (no conversion needed)
            common_dates = cw3e_daily.index.intersection(obs_prec.index)

            if len(common_dates) > 30:
                cw3e_prec = cw3e_daily.loc[common_dates, 'precip_mm']
                snotel_p = obs_prec.loc[common_dates]

                results['prec_bias_mm'] = (cw3e_prec - snotel_p).mean()
                results['prec_mae_mm'] = (cw3e_prec - snotel_p).abs().mean()

                cw3e_total = cw3e_prec.sum()
                snotel_total = snotel_p.sum()

                if snotel_total > 10:  # Need meaningful precip total
                    results['prec_total_bias_pct'] = (cw3e_total - snotel_total) / snotel_total * 100
                results['prec_cw3e_total_mm'] = cw3e_total
                results['prec_snotel_total_mm'] = snotel_total
                results['prec_n_days'] = len(common_dates)

    return results


def categorize_bias(temp_bias: float, prec_bias_pct: float) -> str:
    """Categorize site by bias type."""
    temp_bad = abs(temp_bias) >= TEMP_BIAS_THRESHOLD if not pd.isna(temp_bias) else False
    prec_bad = abs(prec_bias_pct) >= PRECIP_BIAS_THRESHOLD if not pd.isna(prec_bias_pct) else False

    if not temp_bad and not prec_bad:
        return "LOW_BIAS"
    elif temp_bad and not prec_bad:
        return "TEMP_BIAS_ONLY"
    elif not temp_bad and prec_bad:
        return "PRECIP_BIAS_ONLY"
    else:
        return "BOTH_BIASES"


def create_plots(df: pd.DataFrame):
    """Create summary plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CW3E Forcing Bias Analysis - All SNOTEL Sites - WY{WATER_YEAR}',
                 fontsize=14, fontweight='bold')

    # 1. Scatter: temp vs precip bias
    ax = axes[0, 0]
    category_colors = {
        'LOW_BIAS': '#2ca02c',
        'TEMP_BIAS_ONLY': '#ff7f0e',
        'PRECIP_BIAS_ONLY': '#9467bd',
        'BOTH_BIASES': '#d62728',
    }

    for cat, color in category_colors.items():
        mask = df['bias_category'] == cat
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'temp_bias_c'],
                       df.loc[mask, 'prec_total_bias_pct'],
                       c=color, label=f"{cat.replace('_', ' ').title()} ({mask.sum()})",
                       alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=PRECIP_BIAS_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-PRECIP_BIAS_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=TEMP_BIAS_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-TEMP_BIAS_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Temperature Bias (°C)')
    ax.set_ylabel('Precipitation Bias (%)')
    ax.set_title('Site Bias Categories')
    ax.legend(loc='upper left', fontsize=8)

    # 2. Histogram of temp bias
    ax = axes[0, 1]
    temp_data = df['temp_bias_c'].dropna()
    ax.hist(temp_data, bins=50, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=temp_data.mean(), color='red', linestyle='--', linewidth=1,
               label=f'Mean: {temp_data.mean():.2f}°C')
    ax.set_xlabel('Temperature Bias (°C)')
    ax.set_ylabel('Number of Sites')
    ax.set_title('Temperature Bias Distribution')
    ax.legend()

    # 3. Histogram of precip bias
    ax = axes[1, 0]
    prec_data = df['prec_total_bias_pct'].dropna()
    prec_data = prec_data[prec_data.abs() < 100]  # Clip outliers for visualization
    ax.hist(prec_data, bins=50, color='#2ca02c', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=prec_data.mean(), color='red', linestyle='--', linewidth=1,
               label=f'Mean: {prec_data.mean():.1f}%')
    ax.set_xlabel('Precipitation Bias (%)')
    ax.set_ylabel('Number of Sites')
    ax.set_title('Precipitation Bias Distribution')
    ax.legend()

    # 4. Bias vs elevation
    ax = axes[1, 1]
    valid = df[['elevation_m', 'temp_bias_c', 'prec_total_bias_pct']].dropna()
    ax.scatter(valid['elevation_m'], valid['prec_total_bias_pct'],
               c='#1f77b4', alpha=0.5, s=20, label='Precip bias')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Precipitation Bias (%)', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')

    ax2 = ax.twinx()
    ax2.scatter(valid['elevation_m'], valid['temp_bias_c'],
                c='#d62728', alpha=0.5, s=20, marker='s', label='Temp bias')
    ax2.set_ylabel('Temperature Bias (°C)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax.set_title('Bias vs Elevation')

    plt.tight_layout()
    plot_path = FIGURES_DIR / f"forcing_bias_wy{WATER_YEAR}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def generate_markdown(df: pd.DataFrame) -> str:
    """Generate markdown summary report."""
    lines = []

    lines.append(f"# CW3E Forcing Bias Analysis - Water Year {WATER_YEAR}")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Total SNOTEL sites analyzed:** {len(df)}")
    lines.append(f"**Sites with temperature data:** {df['temp_bias_c'].notna().sum()}")
    lines.append(f"**Sites with precipitation data:** {df['prec_total_bias_pct'].notna().sum()}")

    # Overall stats
    lines.append("\n## Overall Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean Temperature Bias | {df['temp_bias_c'].mean():+.2f}°C |")
    lines.append(f"| Temperature MAE | {df['temp_mae_c'].mean():.2f}°C |")
    lines.append(f"| Temperature RMSE | {df['temp_rmse_c'].mean():.2f}°C |")
    lines.append(f"| Mean Precipitation Bias | {df['prec_total_bias_pct'].mean():+.1f}% |")

    # Category counts
    lines.append("\n## Bias Categories\n")
    lines.append(f"**Thresholds:** |T bias| < {TEMP_BIAS_THRESHOLD}°C, |P bias| < {PRECIP_BIAS_THRESHOLD}%\n")

    cat_counts = df['bias_category'].value_counts()
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")
    total = len(df)
    for cat in ['LOW_BIAS', 'TEMP_BIAS_ONLY', 'PRECIP_BIAS_ONLY', 'BOTH_BIASES']:
        count = cat_counts.get(cat, 0)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"| {cat.replace('_', ' ').title()} | {count} | {pct:.1f}% |")

    # State summary
    if 'state' in df.columns:
        lines.append("\n## Summary by State\n")
        state_summary = df.groupby('state').agg({
            'temp_bias_c': 'mean',
            'prec_total_bias_pct': 'mean',
            'site_id': 'count'
        }).round(2)
        state_summary.columns = ['Mean T Bias (°C)', 'Mean P Bias (%)', 'N Sites']
        state_summary = state_summary.sort_values('N Sites', ascending=False)

        lines.append("| State | N Sites | Mean T Bias (°C) | Mean P Bias (%) |")
        lines.append("|-------|---------|----------------:|---------------:|")
        for state, row in state_summary.iterrows():
            lines.append(f"| {state} | {row['N Sites']:.0f} | {row['Mean T Bias (°C)']:+.2f} | {row['Mean P Bias (%)']:+.1f} |")

    # Low bias sites (best for model evaluation)
    lines.append("\n## Low Bias Sites (Recommended for Model Evaluation)\n")
    low_bias = df[df['bias_category'] == 'LOW_BIAS'].sort_values('temp_mae_c')

    if len(low_bias) > 0:
        lines.append(f"**{len(low_bias)} sites** with |T bias| < {TEMP_BIAS_THRESHOLD}°C and |P bias| < {PRECIP_BIAS_THRESHOLD}%\n")
        lines.append("| Site | State | Elev (m) | T Bias (°C) | P Bias (%) |")
        lines.append("|------|-------|----------|------------:|----------:|")

        for _, row in low_bias.head(30).iterrows():
            t = f"{row['temp_bias_c']:+.2f}" if not pd.isna(row['temp_bias_c']) else "N/A"
            p = f"{row['prec_total_bias_pct']:+.1f}" if not pd.isna(row['prec_total_bias_pct']) else "N/A"
            lines.append(f"| {row['site_name']} | {row['state']} | {row['elevation_m']:.0f} | {t} | {p} |")

        if len(low_bias) > 30:
            lines.append(f"\n*... and {len(low_bias) - 30} more sites*")
    else:
        lines.append("*No sites meet the low bias criteria*")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print(f"CW3E Forcing Bias Analysis - WY{WATER_YEAR}")
    print("=" * 70)

    # Load site metadata
    if not SNOTEL_METADATA_FILE.exists():
        print(f"ERROR: Metadata not found: {SNOTEL_METADATA_FILE}")
        print("Run fetch_snotel_metadata.py first")
        return

    sites_df = pd.read_csv(SNOTEL_METADATA_FILE)
    sites_dict = sites_df.set_index('site_id').to_dict('index')
    print(f"Loaded {len(sites_df)} sites from metadata")

    # Load SNOTEL observations
    print("\nLoading SNOTEL observations...")
    snotel_temp, snotel_prec = load_snotel_obs()

    if snotel_temp is not None:
        print(f"  Temperature data: {len(snotel_temp.columns)-1} sites")
    if snotel_prec is not None:
        print(f"  Precipitation data: {len(snotel_prec.columns)-1} sites")

    # Analyze each site
    print("\nAnalyzing sites...")
    all_results = []
    n_processed = 0

    for site_id in sites_df['site_id']:
        cw3e = load_forcing(site_id)
        if cw3e is None:
            continue

        site_info = sites_dict.get(site_id, {})
        result = analyze_site(site_id, site_info, cw3e, snotel_temp, snotel_prec)

        if result.get('temp_bias_c') is not None or result.get('prec_total_bias_pct') is not None:
            all_results.append(result)

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  Processed {n_processed} sites...")

    print(f"\nAnalyzed {len(all_results)} sites with valid comparisons")

    if not all_results:
        print("ERROR: No valid results!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Add categories
    df['bias_category'] = df.apply(
        lambda row: categorize_bias(row.get('temp_bias_c', np.nan),
                                    row.get('prec_total_bias_pct', np.nan)),
        axis=1
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FORCING_BIAS_RESULTS, index=False)
    print(f"\nSaved: {FORCING_BIAS_RESULTS}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sites analyzed: {len(df)}")
    print(f"\nTemperature Bias: {df['temp_bias_c'].mean():+.2f}°C (MAE: {df['temp_mae_c'].mean():.2f}°C)")
    print(f"Precipitation Bias: {df['prec_total_bias_pct'].mean():+.1f}%")

    print("\nBy Category:")
    cat_counts = df['bias_category'].value_counts()
    for cat in ['LOW_BIAS', 'TEMP_BIAS_ONLY', 'PRECIP_BIAS_ONLY', 'BOTH_BIASES']:
        count = cat_counts.get(cat, 0)
        print(f"  {cat.replace('_', ' ').title()}: {count} ({count/len(df)*100:.1f}%)")

    # Create plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    create_plots(df)

    # Generate markdown
    md_content = generate_markdown(df)
    with open(FORCING_BIAS_SUMMARY, 'w') as f:
        f.write(md_content)
    print(f"Saved: {FORCING_BIAS_SUMMARY}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
