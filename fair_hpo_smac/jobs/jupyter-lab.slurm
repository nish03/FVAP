#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --mem=50G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="jupyter-lab"
#SBATCH --output=slurm-%x.out
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=p_discoret

module load \
  modenv/hiera \
  GCC/10.2.0 \
  CUDA/11.1.1 \
  OpenMPI/4.0.5 \
  Python/3.8.6

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

cd ~/prog/discoret/fair_hpo_smac

source env/bin/activate

srun jupyter lab
