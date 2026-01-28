#!/usr/bin/env python3
"""
Fetch CW3E forcing data for WY2003 for all SNOTEL sites.

Reads site list from snotel_metadata.csv and fetches hourly forcing data
from HydroData for each site.

Usage:
    cd ~/Projects/wy2003_forcing_bias/scripts
    source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools
    python fetch_forcing_wy2003.py

    # Fetch specific site only
    python fetch_forcing_wy2003.py --site "Css Lab"

    # Force overwrite existing files
    python fetch_forcing_wy2003.py --force
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import hf_hydrodata as hf
import subsettools as st

from config import (
    HF_EMAIL, HF_PIN,
    WATER_YEAR, START_DATE, END_DATE,
    CW3E_DATASET, CW3E_VERSION,
    FORCING_DIR, SNOTEL_METADATA_FILE,
)


def fetch_forcing_for_site(site_id: str, lat: float, lon: float, verbose: bool = True) -> np.ndarray:
    """
    Fetch CW3E forcing data for a single site.

    Returns array of shape (n_hours, 8) with columns:
    [sw_down, lw_down, precip, t_air, u_wind, v_wind, pressure, q_air]
    """
    try:
        if verbose:
            print(f"    Location: ({lat:.4f}, {lon:.4f})")
            print(f"    Date range: {START_DATE} to {END_DATE}")

        # Get grid bounds for single point
        latlon_bounds = [[lat, lon], [lat, lon]]
        bounds, _ = st.define_latlon_domain(latlon_bounds=latlon_bounds, grid="conus2")

        def fetch_var(varname):
            options = {
                "dataset": CW3E_DATASET,
                "period": "hourly",
                "variable": varname,
                "start_time": START_DATE,
                "end_time": f"{WATER_YEAR}-10-01",  # Include full last day
                "grid_bounds": bounds,
                "dataset_version": CW3E_VERSION
            }
            return hf.get_gridded_data(options)

        if verbose:
            print("    Fetching: SW, LW, Precip, Temp, Wind, Press, Humidity...")

        DSWR = fetch_var("downward_shortwave")
        DLWR = fetch_var("downward_longwave")
        APCP = fetch_var("precipitation")
        Temp = fetch_var("air_temp")
        UGRD = fetch_var("east_windspeed")
        VGRD = fetch_var("north_windspeed")
        Press = fetch_var("atmospheric_pressure")
        SPFH = fetch_var("specific_humidity")

        n_hours = len(DSWR)
        if verbose:
            print(f"    Retrieved {n_hours} hours")

        # Clean precipitation
        APCP[APCP < 0] = 0.0

        # Assemble forcing array
        forcing = np.zeros((n_hours, 8))
        forcing[:, 0] = DSWR[:, 0, 0]
        forcing[:, 1] = DLWR[:, 0, 0]
        forcing[:, 2] = APCP[:, 0, 0]
        forcing[:, 3] = Temp[:, 0, 0]
        forcing[:, 4] = UGRD[:, 0, 0]
        forcing[:, 5] = VGRD[:, 0, 0]
        forcing[:, 6] = Press[:, 0, 0]
        forcing[:, 7] = SPFH[:, 0, 0]

        return forcing

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def save_forcing(forcing: np.ndarray, filepath: Path) -> bool:
    """Save forcing array to text file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for row in forcing:
                line = " ".join(f"{val:.8f}" for val in row)
                f.write(line + "\n")
        return True
    except Exception as e:
        print(f"    ERROR saving: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch CW3E forcing for WY2003")
    parser.add_argument("--site", type=str, help="Fetch specific site only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--limit", type=int, help="Limit number of sites (for testing)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Fetching CW3E Forcing for WY{WATER_YEAR}")
    print("=" * 70)

    # Register API
    hf.register_api_pin(HF_EMAIL, HF_PIN)

    # Load site metadata
    if not SNOTEL_METADATA_FILE.exists():
        print(f"ERROR: Site metadata not found: {SNOTEL_METADATA_FILE}")
        print("Run fetch_snotel_metadata.py first")
        return

    sites_df = pd.read_csv(SNOTEL_METADATA_FILE)
    print(f"Loaded {len(sites_df)} sites from metadata")

    # Filter to specific site if requested
    if args.site:
        sites_df = sites_df[sites_df['site_name'].str.lower() == args.site.lower()]
        if len(sites_df) == 0:
            print(f"ERROR: Site '{args.site}' not found")
            return

    # Limit for testing
    if args.limit:
        sites_df = sites_df.head(args.limit)

    print(f"Processing {len(sites_df)} sites")
    print(f"Output directory: {FORCING_DIR}")
    print("=" * 70)

    # Process each site
    success = 0
    failed = []

    for idx, row in sites_df.iterrows():
        site_id = row['site_id']
        site_name = row.get('site_name', site_id)
        lat = row['latitude']
        lon = row['longitude']

        # Clean site name for filename
        safe_name = str(site_id).replace(" ", "_").replace("/", "_")
        filepath = FORCING_DIR / f"forcing_wy2003_{safe_name}.txt"

        print(f"\n[{idx+1}/{len(sites_df)}] {site_name} ({site_id})")

        # Skip if exists
        if filepath.exists() and not args.force:
            print(f"  File exists: {filepath.name}")
            success += 1
            continue

        # Fetch forcing
        forcing = fetch_forcing_for_site(site_id, lat, lon)

        if forcing is not None:
            if save_forcing(forcing, filepath):
                print(f"  Saved: {filepath.name}")
                success += 1
            else:
                failed.append(site_id)
        else:
            failed.append(site_id)

    # Summary
    print("\n" + "=" * 70)
    print(f"Complete: {success}/{len(sites_df)} sites")
    if failed:
        print(f"Failed ({len(failed)}):")
        for s in failed[:20]:
            print(f"  - {s}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")
    print("=" * 70)


if __name__ == "__main__":
    main()
