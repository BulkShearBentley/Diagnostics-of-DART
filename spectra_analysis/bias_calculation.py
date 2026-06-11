#!/usr/bin/env python
"""
This script calculates the DOMAIN-AVERAGED BIAS of temperature,
specific humidity, u wind, and v wind fields between a WRF
experiment and ERA5 on WRF pressure levels.

Bias = mean(WRF - ERA5)

Assumes dimensions:
    level, x, y
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
    description="Calculate domain-averaged bias of WRF experiment against ERA5"
)

parser.add_argument("--start", type=str, required=True)
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--offset", type=int, default=0)

args = parser.parse_args()

experiment_name = args.exp
start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
offset = args.offset

t0 = start_time + timedelta(hours=3 * offset)
init0 = t0.strftime("%Y%m%d%H%M")

# ========================================================
# Directories
# ========================================================

data_dir = (
    "/users/PAS2635/dominicbentley/DAISE_workflow/"
    "DAISE_diagnostics/my_diagnostics/"
    "spectra_analysis/lowpass_filtered_data_output"
)

output_dir = "./bias_data_output"
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

# ========================================================
# Open datasets
# ========================================================

wrf_nc = ncopen(exp_file)
era5_nc = ncopen(era5_file)

# ========================================================
# Pressure levels
# ========================================================

wrf_levels = wrf_nc.variables["level"][:]
era5_levels = era5_nc.variables["level"][:]

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
# Allocate bias arrays
# ========================================================

temp_bias = np.zeros(nlevels)
q_bias    = np.zeros(nlevels)
u_bias    = np.zeros(nlevels)
v_bias    = np.zeros(nlevels)

matched_era5_levels = np.zeros(nlevels)

# ========================================================
# Bias function
# ========================================================

def compute_bias(a, b):

    a = np.array(a, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)

    diff = a - b

    return np.nanmean(diff)

# ========================================================
# Loop through WRF pressure levels
# ========================================================

for k in range(nlevels):

    wrf_level = wrf_levels[k]

    # Find closest ERA5 pressure level
    era5_k = np.argmin(np.abs(era5_levels - wrf_level))

    matched_era5_levels[k] = era5_levels[era5_k]

    # ----------------------------------------------------
    # Compute domain-averaged bias
    # ----------------------------------------------------

    temp_bias[k] = compute_bias(
        wrf_temp[k, :, :],
        era5_temp[era5_k, :, :]
    )

    q_bias[k] = compute_bias(
        wrf_q[k, :, :],
        era5_q[era5_k, :, :]
    )

    u_bias[k] = compute_bias(
        wrf_u[k, :, :],
        era5_u[era5_k, :, :]
    )

    v_bias[k] = compute_bias(
        wrf_v[k, :, :],
        era5_v[era5_k, :, :]
    )

# ========================================================
# Output file
# ========================================================

output_file = os.path.join(
    output_dir,
    f"{experiment_name}_BIAS_{init0}.nc"
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

temp_var = out_nc.createVariable("Temperature_BIAS", "f4", ("level",))
q_var    = out_nc.createVariable("Specific_Humidity_BIAS", "f4", ("level",))
u_var    = out_nc.createVariable("U_wind_BIAS", "f4", ("level",))
v_var    = out_nc.createVariable("V_wind_BIAS", "f4", ("level",))

# Assign data
level_var[:] = wrf_levels
era5_var[:]  = matched_era5_levels

temp_var[:] = temp_bias
q_var[:]    = q_bias
u_var[:]    = u_bias
v_var[:]    = v_bias

# Attributes
out_nc.description = (
    "Domain-averaged bias of WRF filtered experiment against ERA5"
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
