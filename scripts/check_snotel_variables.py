#!/usr/bin/env python3
"""Quick check of available SNOTEL variables in HydroData."""

import hf_hydrodata as hf

hf.register_api_pin("reedmm@princeton.edu", "4321")

# Check what variables are available
print("Checking SNOTEL variables in HydroData...\n")

# Try to get metadata with different variables to see what works
test_vars = [
    'swe', 'snow_water_equivalent',
    'precipitation', 'precip', 'prec', 'prcpsa', 'precipitation_accumulation',
    'air_temp', 'temperature', 'temp', 'tobs', 'tavg', 'tmax', 'tmin',
    'snow_depth', 'snwd',
]

for var in test_vars:
    try:
        df = hf.get_point_metadata(
            dataset="snotel",
            variable=var,
            temporal_resolution="daily",
            aggregation="sod",
        )
        print(f"  {var}: OK - {len(df)} sites")
    except Exception as e:
        err_msg = str(e)[:60]
        print(f"  {var}: FAILED - {err_msg}")
