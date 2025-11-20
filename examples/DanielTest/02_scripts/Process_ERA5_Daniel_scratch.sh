#!/bin/bash
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 0-1:00
#SBATCH --output=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/logs/process_ERA5_%j.out
#SBATCH --error=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/logs/process_ERA5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.j.vanderhoorn@student.tudelft.nl

# ----------------------------
# Load modules on Snellius
# ----------------------------
module load 2023
module load GCCcore/12.3.0
module load NVHPC/24.5-CUDA-12.1.1
module load netCDF/4.9.2-gompi-2023a
module load netCDF-Fortran/4.6.1-gompi-2023a
module load Anaconda3/2024.06-1

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate hurrywave_2

# Run the Python script
python /gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/Extract_ERA5_per_node.py