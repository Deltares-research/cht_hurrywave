#!/bin/bash
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 0-8:00
#SBATCH --output=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/05_postprocessing/YearSims/hurry.out
#SBATCH --error=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/05_postprocessing/YearSims/hurry.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.j.vanderhoorn@student.tudelft.nl

#module load 2024
#module load netCDF/4.9.2-gompi-2024a
#module load netCDF-Fortran/4.6.1-gompi-2024a

module load 2023
module load GCCcore/12.3.0
module load NVHPC/24.5-CUDA-12.1.1
module load netCDF/4.9.2-gompi-2023a
module load netCDF-Fortran/4.6.1-gompi-2023a
module load Anaconda3/2024.06-1
# Set up conda activation
eval "$(conda shell.bash hook)"
conda activate hurrywave_2

homedir=/home/joseaaa
export PATH=$PATH:/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/06_executables/GPU
#export PYTHONPATH=$PYTHONPATH:$homedir/josseaaa/mypymodules

# hurrywavedir=/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/04_modelruns/YearSims

python /gpfs/work3/0/ai4nbs/hurry_wave/north_sea/02_scripts/Write_Stats_YearSims.py