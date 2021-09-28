#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="jupyter-lab"
#SBATCH --output=slurm-%x.out
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=FAIL
#SBATCH --account=p_discoret

cd ~/prog/discoret/fair_hpo_smac

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

source env/bin/activate

srun jupyter lab
