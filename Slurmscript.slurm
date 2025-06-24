#!/bin/bash
#SBATCH --job-name=my_pytorch_job
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=stellar-gpu

module purge
module load anaconda3/2024.10   # Replace with your version
conda activate base             # Or your conda environment name

python /scratch/gpfs/sl4318/working_dir/ModelScoring
