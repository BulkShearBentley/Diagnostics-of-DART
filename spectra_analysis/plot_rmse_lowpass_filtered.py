#!/usr/bin/env python3

import os
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# ============================================================
# Configuration
# ============================================================

experiments = [
    "NoDA_correct",
    "CTRL",
    #"CTRL_6",
    "NoCloudUpdate",
    #"NoCloudUpdate_6"
]

variables = {
    "Temperature_RMSE": {
        "label": "Temperature RMSE",
        "units": "K"
    },
    "Specific_Humidity_RMSE": {
        "label": "Specific Humidity RMSE",
        "units": "kg kg$^{-1}$"
    },
    "U_wind_RMSE": {
        "label": "U Wind RMSE",
        "units": "m s$^{-1}$"
    },
    "V_wind_RMSE": {
        "label": "V Wind RMSE",
        "units": "m s$^{-1}$"
    }
}

colors = {
    "NoDA_correct": "black",
    "CTRL": "blue",
    #"CTRL_6": "red",
    "NoCloudUpdate": "green",
    #"NoCloudUpdate_6": "purple"
}

output_dir = "rmse_timeseries_plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Read all experiment files
# ============================================================

data = {}

for exp in experiments:

    data_dir = "./rmse_data_output"

    pattern = os.path.join(
        data_dir,
        f"{exp}_RMSE_*.nc"
    )

    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"WARNING: No files found for {exp}")
        continue

    times = []

    # initialize storage
    exp_data = {
        "times": []
    }

    for var in variables:
        exp_data[var] = []

    for file in files:

        # ----------------------------------------------------
        # Extract timestamp from filename
        # ----------------------------------------------------
        # Example:
        # CTRL_3_RMSE_202301130000.nc
        # timestamp = 202301130000

        timestamp_str = file.split("_RMSE_")[-1].replace(".nc", "")

        dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M")

        # ----------------------------------------------------
        # Read netCDF
        # ----------------------------------------------------

        nc = Dataset(file)

        levels = nc.variables["WRF_level"][:]

        # Save levels once
        if "levels" not in exp_data:
            exp_data["levels"] = levels

        exp_data["times"].append(dt)

        for var in variables:
            exp_data[var].append(nc.variables[var][:])

        nc.close()

    # Convert lists to arrays
    for var in variables:
        exp_data[var] = np.array(exp_data[var])

    data[exp] = exp_data

# ============================================================
# Plot one figure per pressure level
# ============================================================

first_exp = list(data.keys())[0]
pressure_levels = data[first_exp]["levels"]

for ilev, pressure in enumerate(pressure_levels):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, var in zip(axes, variables.keys()):

        for exp in experiments:

            if exp not in data:
                continue

            times = data[exp]["times"]

            # Shape:
            # (ntime, nlevel)
            rmse = data[exp][var][:, ilev]

            ax.plot(
                times,
                rmse,
                marker='o',
                linewidth=2,
                label=exp,
                color=colors[exp]
            )

        ax.set_title(variables[var]["label"])

        ax.set_ylabel(variables[var]["units"])

        ax.grid(True, linestyle='--', alpha=0.5)

        ax.tick_params(axis='x', rotation=45)

    axes[0].legend()

    fig.suptitle(
        f"RMSE Time Series at Pressure Level = {pressure/100:.0f} hPa",
        fontsize=16
    )

    plt.tight_layout()

    output_file = os.path.join(
        output_dir,
        f"RMSE_timeseries_{int(pressure/100)}hPa.png"
    )

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"Saved: {output_file}")

print("Done.")
