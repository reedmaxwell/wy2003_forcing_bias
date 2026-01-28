#!/usr/bin/env python3
"""
Compare WY2003 bias results with snow_model_sensitivity sites.
Also create scatter plots of OBS vs forcing for temp and precip.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add snow_model_sensitivity to path to import config
sys.path.insert(0, str(Path.home() / "Projects" / "snow_model_sensitivity"))
from config import SNOTEL_SITES

# Paths
WY2003_DIR = Path.home() / "Projects" / "wy2003_forcing_bias"
RESULTS_DIR = WY2003_DIR / "results"
DATA_DIR = WY2003_DIR / "data"

def main():
    # Load WY2003 bias results
    bias_df = pd.read_csv(RESULTS_DIR / "forcing_bias_wy2003.csv")
    print(f"Loaded {len(bias_df)} sites from WY2003 analysis\n")

    # Extract triplet IDs from snow_model_sensitivity sites
    sensitivity_triplets = {}
    for name, info in SNOTEL_SITES.items():
        triplet = info.get("triplet")
        if triplet:
            sensitivity_triplets[triplet] = {
                "name": name,
                "region": info.get("region"),
                "climate": info.get("climate"),
                "elev_m": info.get("elev_m")
            }

    print(f"Snow model sensitivity study has {len(sensitivity_triplets)} sites with triplets\n")

    # Match sites
    matched = []
    for triplet, site_info in sensitivity_triplets.items():
        match = bias_df[bias_df["site_id"] == triplet]
        if len(match) == 1:
            row = match.iloc[0]
            matched.append({
                "sensitivity_name": site_info["name"],
                "region": site_info["region"],
                "climate": site_info["climate"],
                "site_id": triplet,
                "wy2003_name": row["site_name"],
                "state": row["state"],
                "elev_m": row["elevation_m"],
                "temp_bias_c": row["temp_bias_c"],
                "prec_bias_pct": row["prec_total_bias_pct"],
                "bias_category": row["bias_category"]
            })
        else:
            matched.append({
                "sensitivity_name": site_info["name"],
                "region": site_info["region"],
                "climate": site_info["climate"],
                "site_id": triplet,
                "wy2003_name": "NOT FOUND",
                "state": "N/A",
                "elev_m": site_info["elev_m"],
                "temp_bias_c": np.nan,
                "prec_bias_pct": np.nan,
                "bias_category": "NO_DATA"
            })

    matched_df = pd.DataFrame(matched)

    # Print comparison table
    print("=" * 90)
    print("SNOW MODEL SENSITIVITY SITES - WY2003 BIAS COMPARISON")
    print("=" * 90)
    print(f"{'Site':<20} {'Region':<18} {'State':<6} {'T Bias (°C)':<12} {'P Bias (%)':<12} {'Category'}")
    print("-" * 90)

    for _, row in matched_df.sort_values("region").iterrows():
        t_bias = f"{row['temp_bias_c']:+.2f}" if pd.notna(row['temp_bias_c']) else "N/A"
        p_bias = f"{row['prec_bias_pct']:+.1f}" if pd.notna(row['prec_bias_pct']) else "N/A"
        print(f"{row['sensitivity_name']:<20} {row['region']:<18} {row['state']:<6} {t_bias:<12} {p_bias:<12} {row['bias_category']}")

    print("-" * 90)

    # Summary stats for matched sites
    valid_temp = matched_df["temp_bias_c"].dropna()
    valid_prec = matched_df["prec_bias_pct"].dropna()

    print(f"\nSummary for {len(matched_df)} sensitivity study sites:")
    print(f"  Sites with WY2003 data: {len(valid_temp)} temp, {len(valid_prec)} precip")
    if len(valid_temp) > 0:
        print(f"  Mean temp bias: {valid_temp.mean():+.2f}°C (range: {valid_temp.min():+.2f} to {valid_temp.max():+.2f})")
    if len(valid_prec) > 0:
        print(f"  Mean precip bias: {valid_prec.mean():+.1f}% (range: {valid_prec.min():+.1f}% to {valid_prec.max():+.1f}%)")

    # Count by category
    print(f"\nBias categories:")
    for cat in ["LOW_BIAS", "TEMP_BIAS_ONLY", "PRECIP_BIAS_ONLY", "BOTH_BIASES", "NO_DATA"]:
        count = (matched_df["bias_category"] == cat).sum()
        if count > 0:
            print(f"  {cat}: {count}")

    # Save matched results
    matched_df.to_csv(RESULTS_DIR / "sensitivity_sites_wy2003_bias.csv", index=False)
    print(f"\nSaved matched results to: {RESULTS_DIR / 'sensitivity_sites_wy2003_bias.csv'}")

    # =========================================================================
    # SCATTER PLOTS: OBS vs Forcing
    # =========================================================================
    print("\n" + "=" * 90)
    print("CREATING SCATTER PLOTS")
    print("=" * 90)

    # Load observation data
    temp_obs = pd.read_csv(DATA_DIR / "observations" / "snotel_temp_wy2003.csv")
    prec_obs = pd.read_csv(DATA_DIR / "observations" / "snotel_prec_wy2003.csv")

    print(f"Loaded observations: {len(temp_obs)} temp records, {len(prec_obs)} precip records")

    # Load forcing data and aggregate to daily means
    # We need to compute daily means from hourly forcing for comparison
    # For now, let's use the bias results which already have the comparison

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter for valid data
    valid_data = bias_df.dropna(subset=["temp_bias_c", "prec_total_bias_pct"])

    # -------------------------------------------------------------------------
    # Plot 1: Temperature scatter (using bias to back-calculate)
    # We have: bias = CW3E - SNOTEL, so SNOTEL = CW3E - bias
    # But we don't have the actual values stored, so let's plot bias vs elevation
    # Actually, let's make proper scatter plots using the daily data
    # -------------------------------------------------------------------------

    # For a proper OBS vs Forcing scatter, we need daily data
    # Let's aggregate and merge

    # Temperature: compute site-level stats
    ax1 = axes[0]

    # Use the total precip values we have in bias_df
    valid_prec = bias_df.dropna(subset=["prec_cw3e_total_mm", "prec_snotel_total_mm"])

    # Scatter: SNOTEL total vs CW3E total precipitation
    ax1_data = valid_prec[["prec_snotel_total_mm", "prec_cw3e_total_mm", "state"]].copy()

    # Color by state for interest
    states = ax1_data["state"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(states)))
    state_colors = dict(zip(states, colors))

    for state in states:
        subset = ax1_data[ax1_data["state"] == state]
        ax1.scatter(subset["prec_snotel_total_mm"], subset["prec_cw3e_total_mm"],
                   alpha=0.6, s=30, label=state, c=[state_colors[state]])

    # 1:1 line
    max_val = max(ax1_data["prec_snotel_total_mm"].max(), ax1_data["prec_cw3e_total_mm"].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1 line')

    ax1.set_xlabel("SNOTEL Total Precip (mm)", fontsize=12)
    ax1.set_ylabel("CW3E Total Precip (mm)", fontsize=12)
    ax1.set_title("WY2003 Precipitation: SNOTEL vs CW3E Forcing", fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_xlim(0, max_val * 1.05)
    ax1.set_ylim(0, max_val * 1.05)
    ax1.grid(True, alpha=0.3)

    # Add bias annotation
    mean_bias = valid_prec["prec_total_bias_pct"].mean()
    ax1.text(0.95, 0.05, f"Mean bias: {mean_bias:+.1f}%",
             transform=ax1.transAxes, ha='right', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # -------------------------------------------------------------------------
    # Plot 2: Temperature bias vs elevation (since we don't have raw daily values easily)
    # Actually let's make a histogram of biases instead
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    valid_temp = bias_df.dropna(subset=["temp_bias_c"])

    # Temperature bias histogram
    ax2.hist(valid_temp["temp_bias_c"], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero bias')
    ax2.axvline(x=valid_temp["temp_bias_c"].mean(), color='orange', linestyle='-', lw=2,
                label=f'Mean: {valid_temp["temp_bias_c"].mean():+.2f}°C')

    ax2.set_xlabel("Temperature Bias (CW3E - SNOTEL) [°C]", fontsize=12)
    ax2.set_ylabel("Number of Sites", fontsize=12)
    ax2.set_title("WY2003 Temperature Bias Distribution", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = RESULTS_DIR / "figures" / "obs_vs_forcing_scatter.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scatter plot to: {fig_path}")

    # =========================================================================
    # BONUS: Proper OBS vs Forcing scatter for temperature
    # =========================================================================
    # Create a second figure with temp scatter
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Plot temp bias vs elevation
    ax3 = axes2[0]
    valid_both = bias_df.dropna(subset=["temp_bias_c", "elevation_m"])

    scatter = ax3.scatter(valid_both["elevation_m"], valid_both["temp_bias_c"],
                         c=valid_both["prec_total_bias_pct"], cmap='RdYlBu_r',
                         alpha=0.6, s=30, vmin=-80, vmax=20)
    ax3.axhline(y=0, color='black', linestyle='--', lw=1)
    ax3.axhline(y=1, color='red', linestyle=':', lw=1, alpha=0.7)
    ax3.axhline(y=-1, color='red', linestyle=':', lw=1, alpha=0.7)

    ax3.set_xlabel("Elevation (m)", fontsize=12)
    ax3.set_ylabel("Temperature Bias (CW3E - SNOTEL) [°C]", fontsize=12)
    ax3.set_title("Temperature Bias vs Elevation", fontsize=14)
    ax3.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Precip Bias (%)", fontsize=10)

    # Plot precip bias vs elevation
    ax4 = axes2[1]
    valid_both = bias_df.dropna(subset=["prec_total_bias_pct", "elevation_m"])

    scatter2 = ax4.scatter(valid_both["elevation_m"], valid_both["prec_total_bias_pct"],
                          c=valid_both["temp_bias_c"], cmap='RdYlBu_r',
                          alpha=0.6, s=30, vmin=-3, vmax=3)
    ax4.axhline(y=0, color='black', linestyle='--', lw=1)
    ax4.axhline(y=-10, color='red', linestyle=':', lw=1, alpha=0.7)
    ax4.axhline(y=10, color='red', linestyle=':', lw=1, alpha=0.7)

    ax4.set_xlabel("Elevation (m)", fontsize=12)
    ax4.set_ylabel("Precipitation Bias (%)", fontsize=12)
    ax4.set_title("Precipitation Bias vs Elevation", fontsize=14)
    ax4.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label("Temp Bias (°C)", fontsize=10)

    plt.tight_layout()

    fig2_path = RESULTS_DIR / "figures" / "bias_vs_elevation.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved elevation plot to: {fig2_path}")

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
