"""
Configuration for WY2003 All-SNOTEL Forcing Bias Analysis.

This project compares CW3E atmospheric forcing against SNOTEL observations
for ALL SNOTEL sites for Water Year 2003.
"""
import os
from pathlib import Path

# =============================================================================
# Project Structure
# =============================================================================
PROJECT_ROOT = Path(__file__).parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
FORCING_DIR = DATA_DIR / "forcing"
OBS_DIR = DATA_DIR / "observations"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# =============================================================================
# Water Year Configuration
# =============================================================================
WATER_YEAR = 2003
START_DATE = "2002-10-01"
END_DATE = "2003-09-30"

# =============================================================================
# HydroData Configuration
# =============================================================================
# Set these environment variables or replace with your credentials:
#   export HF_EMAIL="your_email@example.com"
#   export HF_PIN="your_pin"
# Register at: https://hydrogen.princeton.edu/
HF_EMAIL = os.environ.get("HF_EMAIL", "")
HF_PIN = os.environ.get("HF_PIN", "")

if not HF_EMAIL or not HF_PIN:
    import warnings
    warnings.warn(
        "HydroData credentials not set. Set HF_EMAIL and HF_PIN environment variables "
        "or edit config.py directly. Register at https://hydrogen.princeton.edu/"
    )

# SNOTEL dataset parameters
SNOTEL_DATASET = "snotel"
SNOTEL_TEMPORAL = "daily"
SNOTEL_AGGREGATION = "sod"  # start of day

# CW3E dataset parameters
CW3E_DATASET = "CW3E"
CW3E_VERSION = "1.0"

# =============================================================================
# Bias Thresholds
# =============================================================================
TEMP_BIAS_THRESHOLD = 1.0    # °C - absolute value
PRECIP_BIAS_THRESHOLD = 10.0  # % - absolute value

# =============================================================================
# Output Files
# =============================================================================
SNOTEL_METADATA_FILE = DATA_DIR / "snotel_metadata.csv"
FORCING_BIAS_RESULTS = RESULTS_DIR / "forcing_bias_wy2003.csv"
FORCING_BIAS_SUMMARY = RESULTS_DIR / "forcing_bias_wy2003_summary.md"
