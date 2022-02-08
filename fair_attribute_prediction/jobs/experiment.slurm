#!/bin/bash -l

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=32G
#SBATCH --output=slurm-%x-%A-%a.out
#SBATCH --error=slurm-%x-%A-%a.err

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

source env/bin/activate

srun main.py +experiments/args/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}.args --experiment_name "${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}"
