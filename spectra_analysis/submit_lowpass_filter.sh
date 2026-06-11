#!/bin/bash
#SBATCH -A PAS2635
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH -n 10
#SBATCH -J sat_pipeline
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err
#SBATCH --array=0-23


set -euo pipefail

START_TIME="202301130000"

offset=${SLURM_ARRAY_TASK_ID}

EXPERIMENTS=(
    "NoDA_correct"
    #"CTRL_6"
    "CTRL"
    "NoCloudUpdate"
    #"NoCloudUpdate_6"
)

echo "========================================="
echo "OFFSET: ${offset}"
echo "========================================="

# -------------------------------------------------
# 1. Interpolation (all experiments)
# -------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    echo "Interpolating: ${exp}"
    python interpolate_satellite_variables.py \
        --exp "${exp}" \
        --start "${START_TIME}" \
        --offset "${offset}"
done

# -------------------------------------------------
# 2. Lowpass filters
# -------------------------------------------------
echo "Lowpass filter"
python lowpass_filter.py \
    --start "${START_TIME}" \
    --offset "${offset}"

python lowpass_filter_era5.py \
    --start "${START_TIME}" \
    --offset "${offset}"

# -------------------------------------------------
# 3. RMSE
# -------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    echo "RMSE: ${exp}"
    python rmse_calculation.py \
        --exp "${exp}" \
        --start "${START_TIME}" \
        --offset "${offset}"
done

echo "DONE offset ${offset}"

#---------------------------------------------------
# 4. Biases
#---------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    echo "Bias: ${exp}"
    python bias_calculation.py \
        --exp "${exp}" \
        --start "${START_TIME}" \
        --offset "${offset}"
done

echo "DONE offset ${offset}"
