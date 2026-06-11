#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as ncopen
from datetime import datetime, timedelta
import argparse
import os

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Apply lowpass filter to interpolated WRF files based on ERA5 spectra"
)

parser.add_argument(
    "--start",
    type=str,
    required=True,
    help="YYYYMMDDHHMM"
)

parser.add_argument(
    "--member",
    type=int,
    default=5,
    help="Ensemble member index"
)

parser.add_argument("--offset", type=int, default=0)

args = parser.parse_args()

member = args.member

start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
offset=args.offset

t0=start_time+timedelta(hours=3*offset)
init0 =t0.strftime("%Y%m%d%H%M")


# ============================================================
# Directories
# ============================================================

exp_dir = (
    "/users/PAS2635/dominicbentley/DAISE_workflow/"
    "DAISE_diagnostics/my_diagnostics/spectra_analysis/interp_output"
)

era5_dir = (
    "/users/PAS2635/dominicbentley/DAISE_workflow/"
    "DAISE_expt_setup/setup_era5_ens/ens_met_em_files/00001"
)

output_dir = "./lowpass_filtered_data_output"

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# File paths
# ============================================================

# CTRL_6_file = os.path.join(
#     exp_dir,
#     f"interp_var_expt_CTRL_6_{init0}.nc"
# )

CTRL_3_file = os.path.join(
    exp_dir,
    f"interp_var_expt_CTRL_3_{init0}.nc"
)

NoDA_file = os.path.join(
    exp_dir,
    f"interp_var_expt_NoDA_correct_{init0}.nc"
)

# NoCloudUpdate_6_file = os.path.join(
#     exp_dir,
#     f"interp_var_expt_NoCloudUpdate_6_{init0}.nc"
# )

NoCloudUpdate_3_file = os.path.join(
    exp_dir,
    f"interp_var_expt_NoCloudUpdate_3_{init0}.nc"
)


print("\n===================================================")
print(f"WRF FILE : {NoDA_file}")
print("===================================================\n")

# ============================================================
# Check files
# ============================================================

if not os.path.exists(NoDA_file):
    raise FileNotFoundError(f"Missing WRF file:\n{NoDA_file}")

# ============================================================
# Open datasets
# ============================================================

#CTRL_6_nc = ncopen(CTRL_6_file)
CTRL_3_nc = ncopen(CTRL_3_file)
NoDA_nc = ncopen(NoDA_file)
#NoCloudUpdate_6_nc = ncopen(NoCloudUpdate_6_file)
NoCloudUpdate_3_nc = ncopen(NoCloudUpdate_3_file)

# ============================================================
# Coordinates
# ============================================================

lat = NoDA_nc.variables["Latitude"][:, :]
lon = NoDA_nc.variables["Longitude"][:, :]

# ============================================================
# Grid spacing (meters)
# ============================================================

#user input from the template_wrf_namelist.WRF
dx=9000
dy=9000

# ============================================================
# WRF pressure levels
# ============================================================

level = NoDA_nc.variables["level"][:]

# ============================================================
# Function to read experiment fields
# ============================================================

def get_fields(nc):

    temp = nc.variables["Temperature"][
        member, :, :, :
    ]

    q = nc.variables["Specific_Humidity"][
        member, :, :, :
    ]

    u = nc.variables["U_wind"][
        member, :, :, :
    ]

    v = nc.variables["V_wind"][
        member, :, :, :
    ]

    return {
        "Temperature": temp,
        "Specific_Humidity": q,
        "U_wind": u,
        "V_wind": v
    }

# ============================================================
# Read all experiment fields
# ============================================================

#CTRL_6_fields = get_fields(CTRL_6_nc)
CTRL_3_fields = get_fields(CTRL_3_nc)
NoDA_fields = get_fields(NoDA_nc)
#NoCloudUpdate_6_fields = get_fields(NoCloudUpdate_6_nc)
NoCloudUpdate_3_fields = get_fields(NoCloudUpdate_3_nc)

# ============================================================
# Experiment dictionary
# ============================================================

experiments = {
    #"CTRL_6": CTRL_6_fields,
    "CTRL": CTRL_3_fields,
    "NoDA_correct": NoDA_fields,
    #"NoCloudUpdate_6": NoCloudUpdate_6_fields,
    "NoCloudUpdate": NoCloudUpdate_3_fields,
}

# ============================================================
# Labels
# ============================================================

labels = {
    "Temperature": "Temperature (K)",
    "Specific_Humidity": "Specific Humidity (kg/kg)",
    "U_wind": "U wind (m/s)",
    "V_wind": "V wind (m/s)"
}

# ============================================================
# Spectral helper functions
# ============================================================
def low_pass_filter(field, dx, dy, cutoff_km=200.0):

    cutoff_m = cutoff_km * 1000.0
    k_c = 1.0 / cutoff_m  # cutoff wavenumber (1/m)

    shp = field.shape

    # FFT
    fft_field = np.fft.fft2(field)
    fft_field = np.fft.fftshift(fft_field)

    # Wavenumber grids
    kx = np.fft.fftfreq(shp[-1], d=dx)
    ky = np.fft.fftfreq(shp[-2], d=dy)

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    kxm, kym = np.meshgrid(kx, ky)

    kh = np.sqrt(kxm**2 + kym**2)

    # LOW-PASS: keep only large scales (k <= k_c)
    mask = kh <= k_c

    fft_field_lp = fft_field * mask

    fft_field_lp = np.fft.ifftshift(fft_field_lp)
    field_lp = np.real(np.fft.ifft2(fft_field_lp))

    return field_lp

# ============================================================
# Apply low-pass filter and save output
# ============================================================

def apply_filter_to_3d(field3d, dx, dy, cutoff_km=200.0):
    """
    Apply low-pass filter level-by-level.

    field3d shape:
        (nlevel, ny, nx)
    """

    filtered = np.empty_like(field3d)

    for k in range(field3d.shape[0]):
        filtered[k, :, :] = low_pass_filter(
            field3d[k, :, :],
            dx,
            dy,
            cutoff_km=cutoff_km
        )

    return filtered


# ============================================================
# Loop through all experiments
# ============================================================

for experiment_name in [
    "NoDA_correct",
    "CTRL",
    #"CTRL_6",
    "NoCloudUpdate",
    #"NoCloudUpdate_6"
]:

    print("\n===================================================")
    print(f"Processing {experiment_name}")
    print("===================================================\n")

    # --------------------------------------------------------
    # Get experiment fields
    # --------------------------------------------------------

    fields = experiments[experiment_name]

    # --------------------------------------------------------
    # Apply low-pass filters
    # --------------------------------------------------------

    filtered_temperature = apply_filter_to_3d(
        fields["Temperature"],
        dx,
        dy
    )

    filtered_q = apply_filter_to_3d(
        fields["Specific_Humidity"],
        dx,
        dy
    )

    filtered_u = apply_filter_to_3d(
        fields["U_wind"],
        dx,
        dy
    )

    filtered_v = apply_filter_to_3d(
        fields["V_wind"],
        dx,
        dy
    )

    # --------------------------------------------------------
    # Output file name
    # --------------------------------------------------------

    output_file = os.path.join(
        output_dir,
        f"{experiment_name}_lowpass_filtered_"
        f"{init0}.nc"
    )

    print(f"Writing output:\n{output_file}\n")

    # --------------------------------------------------------
    # Create NetCDF output
    # --------------------------------------------------------

    nc_out = ncopen(output_file, "w", format="NETCDF4")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    nlev = len(level)
    ny, nx = lat.shape

    nc_out.createDimension("level", nlev)
    nc_out.createDimension("y", ny)
    nc_out.createDimension("x", nx)

    # --------------------------------------------------------
    # Coordinate variables
    # --------------------------------------------------------

    level_var = nc_out.createVariable(
        "level",
        "f4",
        ("level",)
    )

    lat_var = nc_out.createVariable(
        "Latitude",
        "f4",
        ("y", "x")
    )

    lon_var = nc_out.createVariable(
        "Longitude",
        "f4",
        ("y", "x")
    )

    # --------------------------------------------------------
    # Data variables
    # --------------------------------------------------------

    temp_var = nc_out.createVariable(
        "Temperature",
        "f4",
        ("level", "y", "x")
    )

    q_var = nc_out.createVariable(
        "Specific_Humidity",
        "f4",
        ("level", "y", "x")
    )

    u_var = nc_out.createVariable(
        "U_wind",
        "f4",
        ("level", "y", "x")
    )

    v_var = nc_out.createVariable(
        "V_wind",
        "f4",
        ("level", "y", "x")
    )

    # --------------------------------------------------------
    # Units
    # --------------------------------------------------------

    level_var.units = "Pa"

    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"

    temp_var.units = "K"
    q_var.units = "kg kg-1"

    u_var.units = "m s-1"
    v_var.units = "m s-1"

    # --------------------------------------------------------
    # Write coordinates
    # --------------------------------------------------------

    level_var[:] = level
    lat_var[:, :] = lat
    lon_var[:, :] = lon

    # --------------------------------------------------------
    # Write filtered fields
    # --------------------------------------------------------

    temp_var[:, :, :] = filtered_temperature
    q_var[:, :, :] = filtered_q

    u_var[:, :, :] = filtered_u
    v_var[:, :, :] = filtered_v

    # --------------------------------------------------------
    # Close file
    # --------------------------------------------------------

    nc_out.close()

    print(f"Finished {experiment_name}")
