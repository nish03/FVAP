#!/bin/bash -l

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --job-name="jupyter-lab"
#SBATCH --output=slurm-%x.out

cd ..

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

source env/bin/activate

srun jupyter lab
