#!/bin/bash -l

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --mem=25G
#SBATCH --job-name="tensorboard"
#SBATCH --output=slurm-%x.out

cd ..

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

source env/bin/activate

srun tensorboard --logdir=experiments --host=0.0.0.0 --port=6006
