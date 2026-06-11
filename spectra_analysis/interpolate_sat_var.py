#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as ncopen
from datetime import datetime, timedelta
import argparse
import os

# =========================
# ARGPARSE
# =========================
parser = argparse.ArgumentParser(
    description="Interpolate WRF variables to pressure levels"
)

parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--start", type=str, required=True,
                    help="Start time YYYYMMDDHHMM")
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="./output")

args = parser.parse_args()

experiment = args.exp
start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
offset = args.offset
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

wrf_dir_template = (
    f"/fs/scratch/PAS2635/dominicbentley/IRDA_real_world/"
    f"expt_{experiment}/"
    f"{{init_time}}/prior_ens/"
)

t0 = start_time + timedelta(hours=3*offset)
init0 = t0.strftime("%Y%m%d%H%M")

print(f"Processing {init0}")

# Pressure levels in Pa
levels = np.arange(800, 199, -100) * 100.0


def interp_to_pressure_levels(wrf_p, wrf_var, targets):

    targets = np.asarray(targets)

    nz, ny, nx = wrf_p.shape

    # Ensure pressure decreases upward
    if np.mean(wrf_p[0]) < np.mean(wrf_p[-1]):
        wrf_p = wrf_p[::-1]
        wrf_var = wrf_var[::-1]

    out = np.full((len(targets), ny, nx), np.nan)

    for l, target in enumerate(targets):

        mask = (
            (wrf_p[:-1] >= target) &
            (wrf_p[1:] < target)
        )

        valid = mask.any(axis=0)

        if not np.any(valid):
            continue

        k_above = np.argmax(mask, axis=0)
        k_below = k_above + 1

        j, i = np.where(valid)

        ka = k_above[j, i]
        kb = k_below[j, i]

        p_above = wrf_p[ka, j, i]
        p_below = wrf_p[kb, j, i]

        var_above = wrf_var[ka, j, i]
        var_below = wrf_var[kb, j, i]

        interp_val = var_above + (
            (target - p_above)
            * (var_below - var_above)
            / (p_below - p_above)
        )

        out[l, j, i] = interp_val

    return out


def read_wrf(dir_path, levels):

    f = os.path.join(dir_path, "wrfinput_d01")

    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")

    with ncopen(f, 'r') as ds:

        P = ds.variables['P'][0]
        PB = ds.variables['PB'][0]

        theta = ds.variables['T'][0] + 300.0

        qv = ds.variables['QVAPOR'][0]

        lat = ds.variables['XLAT'][0]
        lon = ds.variables['XLONG'][0]

        u = ds.variables['U'][0]
        v = ds.variables['V'][0]

    # Destagger winds
    u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v = 0.5 * (v[:, :-1, :] + v[:, 1:, :])

    p = P + PB

    kappa = 0.2854

    T = theta * (p / 100000.0) ** kappa

    q = qv / (1.0 + qv)

    temp_p = interp_to_pressure_levels(p, T, levels)
    q_p = interp_to_pressure_levels(p, q, levels)
    u_p = interp_to_pressure_levels(p, u, levels)
    v_p = interp_to_pressure_levels(p, v, levels)

    return temp_p, q_p, u_p, v_p, lat, lon


all_members = []

for member in range(1, 51):

    member_id = f"{member:05d}"

    member_dir = os.path.join(
        wrf_dir_template.format(init_time=init0),
        member_id
    )

    print(f"Loading member {member_id}")

    try:
        temp, q_2d, u_wind, v_wind, lat, lon = read_wrf(
            member_dir,
            levels
        )

        all_members.append({
            "member": member_id,
            "temp": temp,
            "q": q_2d,
            "u": u_wind,
            "v": v_wind,
            "lat": lat,
            "lon": lon
        })

    except FileNotFoundError as e:
        print(e)
        continue

temp_all = np.array([m["temp"] for m in all_members])
q_all = np.array([m["q"] for m in all_members])
u_all = np.array([m["u"] for m in all_members])
v_all = np.array([m["v"] for m in all_members])

lat = all_members[0]["lat"]
lon = all_members[0]["lon"]

# =========================
# SAVE OUTPUT
# =========================

out_file = os.path.join(
    output_dir,
    f"interp_var_expt_{experiment}_3_{t0.strftime('%Y%m%d%H%M')}.nc"
)

nmember, nlev, ny, nx = temp_all.shape

with ncopen(out_file, 'w') as nc:

    # Dimensions
    nc.createDimension('member', nmember)
    nc.createDimension('level', nlev)
    nc.createDimension('y', ny)
    nc.createDimension('x', nx)

    # Coordinates
    member_var = nc.createVariable('member', 'i4', ('member',))
    member_var[:] = np.arange(1, nmember + 1)

    lev = nc.createVariable('level', 'f4', ('level',))
    lev.units = 'Pa'
    lev[:] = levels

    # Variables
    temp_var = nc.createVariable(
        'Temperature',
        'f4',
        ('member', 'level', 'y', 'x'),
        zlib=True
    )

    q_var = nc.createVariable(
        'Specific_Humidity',
        'f4',
        ('member', 'level', 'y', 'x'),
        zlib=True
    )

    u_var = nc.createVariable(
        'U_wind',
        'f4',
        ('member', 'level', 'y', 'x'),
        zlib=True
    )

    v_var = nc.createVariable(
        'V_wind',
        'f4',
        ('member', 'level', 'y', 'x'),
        zlib=True
    )

    lat_var = nc.createVariable(
        'Latitude',
        'f4',
        ('y', 'x')
    )

    lon_var = nc.createVariable(
        'Longitude',
        'f4',
        ('y', 'x')
    )

    # Write data
    temp_var[:] = temp_all
    q_var[:] = q_all
    u_var[:] = u_all
    v_var[:] = v_all

    lat_var[:] = lat
    lon_var[:] = lon

print(f"Saved: {out_file}")
