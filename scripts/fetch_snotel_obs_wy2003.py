#!/usr/bin/env python3
"""
Fetch SNOTEL temperature and precipitation observations for WY2003.

Retrieves daily observations from HydroData for all SNOTEL sites.

Usage:
    cd ~/Projects/wy2003_forcing_bias/scripts
    source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools
    python fetch_snotel_obs_wy2003.py
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import hf_hydrodata as hf

from config import (
    HF_EMAIL, HF_PIN,
    WATER_YEAR, START_DATE, END_DATE,
    SNOTEL_DATASET, SNOTEL_TEMPORAL, SNOTEL_AGGREGATION,
    OBS_DIR, SNOTEL_METADATA_FILE,
)


def fetch_variable_batched(site_ids: list, variable: str, aggregation: str = None, batch_size: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Fetch a variable for multiple sites in batches to avoid URI too long errors."""
    all_data = []
    n_batches = (len(site_ids) + batch_size - 1) // batch_size

    # Use provided aggregation or fall back to default
    agg = aggregation if aggregation else SNOTEL_AGGREGATION

    if verbose:
        print(f"  Fetching {variable} (agg={agg}) in {n_batches} batches of {batch_size}...")

    for i in range(0, len(site_ids), batch_size):
        batch = site_ids[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            data = hf.get_point_data(
                dataset=SNOTEL_DATASET,
                variable=variable,
                temporal_resolution=SNOTEL_TEMPORAL,
                aggregation=agg,
                site_ids=batch,
                date_start=START_DATE,
                date_end=END_DATE,
            )

            if data is not None and len(data) > 0:
                all_data.append(data)
                if verbose and batch_num % 5 == 0:
                    print(f"    Batch {batch_num}/{n_batches} complete")

        except Exception as e:
            print(f"    Batch {batch_num} error: {e}")
            continue

    if not all_data:
        print(f"  ERROR: No data retrieved for {variable}")
        return None

    # Merge all batches
    if verbose:
        print(f"    Merging {len(all_data)} batches...")

    # First df as base
    merged = all_data[0]

    for df in all_data[1:]:
        # Merge on date, adding new site columns
        merged = merged.merge(df, on='date', how='outer')

    if verbose:
        print(f"    Retrieved data for {len(merged.columns)-1} sites")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch SNOTEL observations for WY2003")
    parser.add_argument("--limit", type=int, help="Limit number of sites (for testing)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Fetching SNOTEL Observations for WY{WATER_YEAR}")
    print("=" * 70)
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Register API
    hf.register_api_pin(HF_EMAIL, HF_PIN)

    # Load site metadata
    if not SNOTEL_METADATA_FILE.exists():
        print(f"ERROR: Site metadata not found: {SNOTEL_METADATA_FILE}")
        print("Run fetch_snotel_metadata.py first")
        return

    sites_df = pd.read_csv(SNOTEL_METADATA_FILE)
    print(f"Loaded {len(sites_df)} sites from metadata")

    # Get site IDs
    site_ids = sites_df['site_id'].tolist()

    if args.limit:
        site_ids = site_ids[:args.limit]
        print(f"Limited to {len(site_ids)} sites for testing")

    # Create output directory
    OBS_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch temperature (air_temp with aggregation="mean")
    print("\n" + "-" * 50)
    print("Fetching Temperature (air_temp, aggregation=mean)...")
    print("-" * 50)

    temp_data = fetch_variable_batched(site_ids, "air_temp", aggregation="mean")

    if temp_data is not None:
        temp_file = OBS_DIR / "snotel_temp_wy2003.csv"
        temp_data.to_csv(temp_file, index=False)
        print(f"  Saved: {temp_file}")
    else:
        print("  WARNING: Could not fetch temperature data")

    # Fetch precipitation (precipitation with aggregation="sum" for daily totals)
    print("\n" + "-" * 50)
    print("Fetching Precipitation (precipitation, aggregation=sum)...")
    print("-" * 50)

    prec_data = fetch_variable_batched(site_ids, "precipitation", aggregation="sum")

    if prec_data is not None:
        prec_file = OBS_DIR / "snotel_prec_wy2003.csv"
        prec_data.to_csv(prec_file, index=False)
        print(f"  Saved: {prec_file}")
    else:
        print("  WARNING: Could not fetch precipitation data")

    # Summary
    print("\n" + "=" * 70)
    print("Fetch complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
