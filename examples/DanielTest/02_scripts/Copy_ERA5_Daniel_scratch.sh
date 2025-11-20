#!/bin/bash
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --array=0-2
#SBATCH -t 0-24:00
#SBATCH --output=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/logs/copy_ERA5_%A_%a.out
#SBATCH --error=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/logs/copy_ERA5_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.j.vanderhoorn@student.tudelft.nl

# Directories
SRC_BASE="/gpfs/work3/0/ai4nbs/ERA5_data/data"
DEST_BASE="/scratch-shared/dvdhoorn/ERA5_data_nonedit"
LOG_DIR="/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/logs"

# Create destination and log directories if they don't exist
mkdir -p "$DEST_BASE"
mkdir -p "$LOG_DIR"

# List of folders to copy (in the same order as array indices)
folders=(
  "mean_wave_direction"
  "peak_wave_period"
  "significant_height_of_combined_wind_waves_and_swell"
)

# Select the folder for this array index
FOLDER=${folders[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Starting copy for $FOLDER (Array ID: $SLURM_ARRAY_TASK_ID)" | tee -a "$LOG_DIR/copy_summary.log"

# Perform the copy with rsync
rsync -avh --progress "$SRC_BASE/$FOLDER/" "$DEST_BASE/$FOLDER/" \
    >> "$LOG_DIR/${FOLDER}_copy.log" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] Copy completed successfully for $FOLDER (Array ID: $SLURM_ARRAY_TASK_ID)" | tee -a "$LOG_DIR/copy_summary.log"
else
    echo "[$(date)] Copy failed for $FOLDER (Array ID: $SLURM_ARRAY_TASK_ID)" | tee -a "$LOG_DIR/copy_summary.log"
fi