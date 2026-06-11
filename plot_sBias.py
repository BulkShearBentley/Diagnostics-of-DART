import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ncopen
from datetime import datetime

'''
USER SETTINGS
'''
input_dir = "/users/PAS2635/dominicbentley/DAISE_workflow/DAISE_diagnostics/my_diagnostics/output"
save_dir = "/users/PAS2635/dominicbentley/DAISE_workflow/DAISE_diagnostics/my_diagnostics/figs"
os.makedirs(save_dir, exist_ok=True)

'''
READ DATA
'''
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".ncz")])

times = []
inc_T_all = []
mod_T_all = []
tot_T_all = []

inc_q_all = []
mod_q_all = []
tot_q_all = []

levels = None

for f in files:
    filepath = os.path.join(input_dir, f)

    with ncopen(filepath, 'r') as nc:
        nc.set_auto_maskandscale(True)  

        # Extract time from filename
        t_str = f.split('_')[1].split('.')[0]
        t = datetime.strptime(t_str, "%Y%m%d%H%M")
        times.append(t)

        # Read variables 
        inc_T = nc.variables['inc_T'][:]
        mod_T = nc.variables['mod_T'][:]
        tot_T = nc.variables['tot_T'][:]

        inc_q = nc.variables['inc_q'][:]
        mod_q = nc.variables['mod_q'][:]
        tot_q = nc.variables['tot_q'][:]

        # Convert masked arrays to NaN arrays
        inc_T = np.ma.filled(inc_T, np.nan)
        mod_T = np.ma.filled(mod_T, np.nan)
        tot_T = np.ma.filled(tot_T, np.nan)

        inc_q = np.ma.filled(inc_q, np.nan)
        mod_q = np.ma.filled(mod_q, np.nan)
        tot_q = np.ma.filled(tot_q, np.nan)

        inc_T_all.append(inc_T)
        mod_T_all.append(mod_T)
        tot_T_all.append(tot_T)

        inc_q_all.append(inc_q)
        mod_q_all.append(mod_q)
        tot_q_all.append(tot_q)

        
        if levels is None:
            levels = nc.variables['level'][:]

# Convert to arrays (time x level)
inc_T_all = np.array(inc_T_all)
mod_T_all = np.array(mod_T_all)
tot_T_all = np.array(tot_T_all)

inc_q_all = np.array(inc_q_all)
mod_q_all = np.array(mod_q_all)
tot_q_all = np.array(tot_q_all)


# REMOVE LEVELS THAT ARE NaN
valid_levels = ~np.all(np.isnan(tot_T_all), axis=0)

levels = levels[valid_levels]
inc_T_all = inc_T_all[:, valid_levels]
mod_T_all = mod_T_all[:, valid_levels]
tot_T_all = tot_T_all[:, valid_levels]

inc_q_all = inc_q_all[:, valid_levels]
mod_q_all = mod_q_all[:, valid_levels]
tot_q_all = tot_q_all[:, valid_levels]







# =========================
# PLOTTING FUNCTION
# =========================
from matplotlib.colors import TwoSlopeNorm

def plot_time_pressure(data, levels, times, title, fname):
    plt.figure(figsize=(10,6))

    # Mask NaNs
    data = np.ma.masked_invalid(data)

    # Robust symmetric limits around 0
    vabs = np.nanpercentile(np.abs(data), 98)

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    t_axis = np.arange(len(times))

    cf = plt.contourf(
        t_axis,
        levels,
        data.T,
        levels=21,
        cmap="RdBu_r",
        norm=norm,
        extend="both"
    )

    plt.gca().invert_yaxis()
    plt.colorbar(cf, label="Normalized Value")

    plt.xticks(
        t_axis[::2],
        [t.strftime("%m-%d\n%H:%M") for t in times][::2]
    )

    plt.ylabel("Pressure (Pa)")
    plt.xlabel("Time")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

# =========================
# MAKE PLOTS
# =========================
plot_time_pressure(inc_T_all, levels, times,
                   "Increment (T)", "inc_T.png")

plot_time_pressure(mod_T_all, levels, times,
                   "Model Response (T)", "mod_T.png")

plot_time_pressure(tot_T_all, levels, times,
                   "Total ΔsBias (T)", "tot_T.png")

plot_time_pressure(inc_q_all, levels, times,
                   "Increment (q)", "inc_q.png")

plot_time_pressure(mod_q_all, levels, times,
                   "Model Response (q)", "mod_q.png")

plot_time_pressure(tot_q_all, levels, times,
                   "Total ΔsBias (q)", "tot_q.png")

print("Done! Plots saved to:", save_dir)
