#!/bin/bash
#SBATCH -A PAS2635
#SBATCH -t 04:00:00
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --array=0-68
#SBATCH -J biasCTRL
#SBATCH -o logs/bias_%A_%a.out
#SBATCH -e logs/bias_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominicbentley

# =========================
# SETTINGS
# =========================
EXP="CTRL"
START="202301130000"
OUTPUT_DIR="./output"
mkdir -p $OUTPUT_DIR

# =========================
# RUN PYTHON
# =========================
python bias_contributions_v2.py \
    --exp $EXP \
    --start $START \
    --offset $SLURM_ARRAY_TASK_ID \
    --output_dir $OUTPUT_DIR
