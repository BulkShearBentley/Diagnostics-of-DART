#!/usr/bin/env python
import numpy as np
from netCDF4 import Dataset as ncopen
from datetime import datetime, timedelta
import argparse
import os

# =========================
# ARGPARSE
# =========================
parser = argparse.ArgumentParser(description="Compute deltasBias (900-200 mb) for one 3h window")
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--start", type=str, required=True, help="Start time in YYYYMMDDHHMM")
parser.add_argument("--offset", type=int, default=0, help="Number of 3-hour steps after start")
parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save .ncz files")
args = parser.parse_args()

experiment = args.exp
start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
offset = args.offset
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# =========================
# SETTINGS
# =========================
ensemble_size = 50

NoDA_dir_template = f"/fs/scratch/PAS2635/dominicbentley/IRDA_real_world/expt_NoDA/{{init_time}}/prior_ens/"
wrf_dir_template = f"/fs/scratch/PAS2635/dominicbentley/IRDA_real_world/expt_{experiment}/{{init_time}}/prior_ens/"
wrf_analysis_template = f"/fs/scratch/PAS2635/dominicbentley/IRDA_real_world/expt_NoDA/{{init_time}}/posterior_ens/"

era5_template = '/users/PAS2635/dominicbentley/DAISE_workflow/DAISE_expt_setup/setup_era5_icbc/wps+real_era5/met_em.d01.{time}.nc'

# =========================
# TIME HANDLING
# =========================
t0 = start_time + timedelta(hours=1 * offset)
t1 = t0 + timedelta(hours=1)

init0 = t0.strftime("%Y%m%d%H%M")
init1 = t1.strftime("%Y%m%d%H%M")
era0 = t0.strftime("%Y-%m-%d_%H:%M:%S")
era1 = t1.strftime("%Y-%m-%d_%H:%M:%S")

print(f"Processing {t0} → {t1}")

# =========================
# FUNCTIONS
# =========================
import numpy as np


def interp_to_pressure_levels(wrf_p, wrf_var, targets):
    targets = np.asarray(targets)
    nz, ny, nx = wrf_p.shape

    # Ensure pressure decreases with height
    if np.mean(wrf_p[0]) < np.mean(wrf_p[-1]):
        wrf_p = wrf_p[::-1]
        wrf_var = wrf_var[::-1]

    nlev = len(targets)
    out = np.full((nlev, ny, nx), np.nan)

    for l, target in enumerate(targets):
        mask = (wrf_p[:-1] >= target) & (wrf_p[1:] < target)
        valid = mask.any(axis=0)

        if not np.any(valid):
            continue

        k_above = np.argmax(mask, axis=0)
        k_below = k_above + 1

        k_above = np.clip(k_above, 0, nz-2)
        k_below = np.clip(k_below, 1, nz-1)

        j, i = np.indices((ny, nx))
        ja = j[valid]
        ia = i[valid]
        ka = k_above[valid]
        kb = k_below[valid]

        p_above = wrf_p[ka, ja, ia]
        p_below = wrf_p[kb, ja, ia]
        var_above = wrf_var[ka, ja, ia]
        var_below = wrf_var[kb, ja, ia]

        interp_val = var_above + (target - p_above) * (
            (var_below - var_above) / (p_below - p_above)
        )

        interp = np.full((ny, nx), np.nan)
        interp[ja, ia] = interp_val
        out[l] = interp

    return out

def horiz_mean(field):
    return np.nanmean(field, axis=(1,2))

def read_era5(time_str):
    fname = era5_template.format(time=time_str)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing ERA5 file: {fname}")

    with ncopen(fname, 'r') as ds:
        levels = ds.variables['PRES'][0,:,0,0]
        T = ds.variables['TT'][0]
        RH = ds.variables['RH'][0]

        if levels[0] < levels[-1]:
            levels = levels[::-1]
            T = T[::-1]
            RH = RH[::-1]

    p3d = levels[:, None, None]
    e_s = 6.112*np.exp(17.67*(T-273.15)/((T-273.15)+243.5))*100
    e = RH/100 * e_s
    q = 0.622*e/(p3d - 0.378*e)

    return levels, T, q

def read_wrf_ensemble(dir_path, levels):
    T_ens, q_ens = [], []

    for mem in range(1, ensemble_size+1):
        f = os.path.join(dir_path, f"{mem:05d}/wrfinput_d01")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing WRF file: {f}")

        with ncopen(f,'r') as ds:
            P = ds.variables['P'][0]
            PB = ds.variables['PB'][0]
            theta = ds.variables['T'][0] + 300
            QV = ds.variables['QVAPOR'][0]

        p = P + PB
        kappa = 0.2854
        T = theta * (p/100000.0)**kappa
        q = QV / (1 + QV)

        T_p = interp_to_pressure_levels(p, T, levels)
        q_p = interp_to_pressure_levels(p, q, levels)

        T_ens.append(T_p)
        q_ens.append(q_p)

    return np.nanmean(T_ens, axis=0), np.nanmean(q_ens, axis=0)

def compute_rmse(noda_dir, levels, era_T, era_q):
    wrf_T, wrf_q = read_wrf_ensemble(noda_dir, levels)
    rmse_T = np.sqrt(np.nanmean((wrf_T - era_T)**2, axis=(1,2)))
    rmse_q = np.sqrt(np.nanmean((wrf_q - era_q)**2, axis=(1,2)))
    rmse_T = np.maximum(rmse_T, 1e-3)
    rmse_q = np.maximum(rmse_q, 1e-8)
    return rmse_T, rmse_q

# =========================
# READ DATA
# =========================
levels_full, era_T0, era_q0 = read_era5(era0)
mask850 = levels_full <= 85000

levels = levels_full[mask850]
era_T0 = era_T0[mask850]
era_q0 = era_q0[mask850]
_, era_T1, era_q1 = read_era5(era1)
era_T1 = era_T1[mask850]
era_q1 = era_q1[mask850]


prior_T0, prior_q0 = read_wrf_ensemble(wrf_dir_template.format(init_time=init0), levels)
analysis_T0, analysis_q0 = read_wrf_ensemble(wrf_analysis_template.format(init_time=init0), levels)
prior_T1, prior_q1 = read_wrf_ensemble(wrf_dir_template.format(init_time=init1), levels)

rmse_T, rmse_q = compute_rmse(NoDA_dir_template.format(init_time=init0), levels, era_T0, era_q0)

# =========================
# COMPUTE BIAS CONTRIBUTIONS
# =========================
pT0 = horiz_mean(prior_T0)
aT0 = horiz_mean(analysis_T0)
pT1 = horiz_mean(prior_T1)
eT0 = horiz_mean(era_T0)
eT1 = horiz_mean(era_T1)

inc_T = (aT0 - pT0) / rmse_T
mod_T = ((pT1 - aT0) - (eT1 - eT0)) / rmse_T
tot_T = inc_T + mod_T

pq0 = horiz_mean(prior_q0)
aq0 = horiz_mean(analysis_q0)
pq1 = horiz_mean(prior_q1)
eq0 = horiz_mean(era_q0)
eq1 = horiz_mean(era_q1)

inc_q = (aq0 - pq0) / rmse_q
mod_q = ((pq1 - aq0) - (eq1 - eq0)) / rmse_q
tot_q = inc_q + mod_q


# =========================
# SAVE TO NETCDF
# =========================
out_file = os.path.join(output_dir, f"deltasBias_{t0.strftime('%Y%m%d%H%M')}.ncz")
with ncopen(out_file, 'w') as nc:
    nc.createDimension('level', len(levels))
    nc.createDimension('time', 1)

    lev = nc.createVariable('level', 'f4', ('level',))
    lev[:] = levels

    nc.createVariable('inc_T', 'f4', ('level',))[:] = inc_T
    nc.createVariable('mod_T', 'f4', ('level',))[:] = mod_T
    nc.createVariable('tot_T', 'f4', ('level',))[:] = tot_T
    nc.createVariable('inc_q', 'f4', ('level',))[:] = inc_q
    nc.createVariable('mod_q', 'f4', ('level',))[:] = mod_q
    nc.createVariable('tot_q', 'f4', ('level',))[:] = tot_q

print(f"Saved: {out_file}")
