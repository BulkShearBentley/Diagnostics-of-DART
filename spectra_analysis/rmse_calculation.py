#!/usr/bin/env python
"""
This script calculates the RMSE of temperature, specific humidity,
u wind, and v wind fields between a WRF experiment and ERA5
on WRF pressure levels.

Assumes dimensions:
    level, x, y

Variables:
    level
    Temperature
    Specific_Humidity
    U_wind
    V_wind

Written by Dominic Bentley
"""

import numpy as np
import os
from datetime import datetime, timedelta
import argparse
from netCDF4 import Dataset as ncopen

# ========================================================
# Arguments
# ========================================================

parser = argparse.ArgumentParser(
    description="Calculate RMSE of WRF experiment against ERA5"
)

parser.add_argument(
    "--start",
    type=str,
    required=True,
    help="YYYYMMDDHHMM"
)

parser.add_argument(
    "--exp",
    type=str,
    required=True,
    help="Experiment name (e.g. NoDA, CTRL)"
)

parser.add_argument("--offset", type=int, default=0)

args = parser.parse_args()

experiment_name = args.exp
start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
offset=args.offset

t0=start_time+timedelta(hours=3*offset)
init0 =t0.strftime("%Y%m%d%H%M")

# ========================================================
# Directories
# ========================================================

data_dir = (
    "/users/PAS2635/dominicbentley/DAISE_workflow/"
    "DAISE_diagnostics/my_diagnostics/"
    "spectra_analysis/lowpass_filtered_data_output"
)

output_dir = "./rmse_data_output"
os.makedirs(output_dir, exist_ok=True)

# ========================================================
# File Paths
# ========================================================

era5_file = os.path.join(
    data_dir,
    f"ERA5_lowpass_filtered_{init0}.nc"
)

exp_file = os.path.join(
    data_dir,
    f"{experiment_name}_lowpass_filtered_{init0}.nc"
)

print("\n===================================================")
print(f"WRF FILE  : {exp_file}")
print(f"ERA5 FILE : {era5_file}")
print("===================================================\n")

# ========================================================
# Check files
# ========================================================

if not os.path.exists(exp_file):
    raise FileNotFoundError(f"Missing WRF file:\n{exp_file}")

if not os.path.exists(era5_file):
    raise FileNotFoundError(f"Missing ERA5 file:\n{era5_file}")

# ========================================================
# Open datasets
# ========================================================

wrf_nc = ncopen(exp_file)
era5_nc = ncopen(era5_file)

# ========================================================
# Pressure levels
# ========================================================

wrf_levels = wrf_nc.variables["level"][:]      # Pa
era5_levels = era5_nc.variables["level"][:]    # Pa

nlevels = len(wrf_levels)

# ========================================================
# Read variables
# ========================================================

wrf_temp = wrf_nc.variables["Temperature"][:]
wrf_q    = wrf_nc.variables["Specific_Humidity"][:]
wrf_u    = wrf_nc.variables["U_wind"][:]
wrf_v    = wrf_nc.variables["V_wind"][:]

era5_temp = era5_nc.variables["Temperature"][:]
era5_q    = era5_nc.variables["Specific_Humidity"][:]
era5_u    = era5_nc.variables["U_wind"][:]
era5_v    = era5_nc.variables["V_wind"][:]

# ========================================================
# Allocate RMSE arrays
# ========================================================

temp_rmse = np.zeros(nlevels)
q_rmse    = np.zeros(nlevels)
u_rmse    = np.zeros(nlevels)
v_rmse    = np.zeros(nlevels)

matched_era5_levels = np.zeros(nlevels)

# ========================================================
# RMSE function
# ========================================================

def compute_rmse(a, b):

    # Convert NetCDF/masked arrays into writable ndarray
    a = np.array(a, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)

    diff2 = (a - b) ** 2

    return np.sqrt(np.nanmean(diff2))

# ========================================================
# Loop through WRF pressure levels
# ========================================================

for k in range(nlevels):

    wrf_level = wrf_levels[k]

    # Find closest ERA5 pressure level
    era5_k = np.argmin(np.abs(era5_levels - wrf_level))

    matched_era5_levels[k] = era5_levels[era5_k]

    # ----------------------------------------------------
    # Compute RMSE
    # ----------------------------------------------------

    temp_rmse[k] = compute_rmse(
        wrf_temp[k, :, :],
        era5_temp[era5_k, :, :]
    )

    q_rmse[k] = compute_rmse(
        wrf_q[k, :, :],
        era5_q[era5_k, :, :]
    )

    u_rmse[k] = compute_rmse(
        wrf_u[k, :, :],
        era5_u[era5_k, :, :]
    )

    v_rmse[k] = compute_rmse(
        wrf_v[k, :, :],
        era5_v[era5_k, :, :]
    )

    print(
        f"WRF Level: {wrf_level:.0f} Pa | "
        f"ERA5 Level: {era5_levels[era5_k]:.0f} Pa"
    )

# ========================================================
# Output file
# ========================================================

output_file = os.path.join(
    output_dir,
    f"{experiment_name}_RMSE_{init0}.nc"
)

print(f"\nWriting output file:\n{output_file}\n")

# ========================================================
# Create output NetCDF
# ========================================================

out_nc = ncopen(output_file, "w")

# Dimensions
out_nc.createDimension("level", nlevels)

# Variables
level_var = out_nc.createVariable("WRF_level", "f4", ("level",))
era5_var  = out_nc.createVariable("ERA5_level", "f4", ("level",))

temp_var = out_nc.createVariable("Temperature_RMSE", "f4", ("level",))
q_var    = out_nc.createVariable("Specific_Humidity_RMSE", "f4", ("level",))
u_var    = out_nc.createVariable("U_wind_RMSE", "f4", ("level",))
v_var    = out_nc.createVariable("V_wind_RMSE", "f4", ("level",))

# Assign data
level_var[:] = wrf_levels
era5_var[:]  = matched_era5_levels

temp_var[:] = temp_rmse
q_var[:]    = q_rmse
u_var[:]    = u_rmse
v_var[:]    = v_rmse

# Attributes
out_nc.description = (
    "RMSE of WRF filtered experiment against ERA5"
)

out_nc.experiment = experiment_name
out_nc.creation_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

level_var.units = "Pa"
era5_var.units = "Pa"

temp_var.units = "K"
q_var.units = "kg kg-1"
u_var.units = "m s-1"
v_var.units = "m s-1"

# Close files
out_nc.close()
wrf_nc.close()
era5_nc.close()

print("Done.")
