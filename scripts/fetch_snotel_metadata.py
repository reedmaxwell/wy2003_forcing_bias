#!/usr/bin/env python3
"""
Fetch metadata for ALL SNOTEL sites from HydroData.

Saves a CSV with site_id, site_name, latitude, longitude, elevation, state, etc.

Usage:
    cd ~/Projects/wy2003_forcing_bias/scripts
    source ~/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools
    python fetch_snotel_metadata.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hf_hydrodata as hf
from config import (
    HF_EMAIL, HF_PIN, DATA_DIR,
    SNOTEL_DATASET, SNOTEL_TEMPORAL, SNOTEL_AGGREGATION,
    SNOTEL_METADATA_FILE,
)


def main():
    print("=" * 70)
    print("Fetching SNOTEL Site Metadata from HydroData")
    print("=" * 70)

    # Register API
    hf.register_api_pin(HF_EMAIL, HF_PIN)

    # Fetch metadata for all SNOTEL sites
    # Using 'swe' as the variable to get snow-reporting sites
    print("\nQuerying HydroData for SNOTEL sites...")

    metadata_df = hf.get_point_metadata(
        dataset=SNOTEL_DATASET,
        variable="swe",
        temporal_resolution=SNOTEL_TEMPORAL,
        aggregation=SNOTEL_AGGREGATION,
    )

    print(f"\nRetrieved {len(metadata_df)} SNOTEL sites")
    print(f"\nColumns: {list(metadata_df.columns)}")

    # Display sample
    print("\nSample sites:")
    print(metadata_df.head(10).to_string())

    # Summary by state (if available)
    if 'state' in metadata_df.columns:
        print("\nSites by state:")
        state_counts = metadata_df['state'].value_counts()
        for state, count in state_counts.items():
            print(f"  {state}: {count}")

    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(SNOTEL_METADATA_FILE, index=False)
    print(f"\nSaved: {SNOTEL_METADATA_FILE}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
