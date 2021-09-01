#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH -J "reconstruction-loss-ms-ssim-baseline"
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_discoret

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

source ~/.virtualenvs/discoret/bin/activate
cd ~/workspaces/ssd/s9911486-discoret
~/prog/discoret/fair_hpo_smac/smac-hpo.py \
  --utkface-dir datasets/UTKFace \
  --output-dir experiments/${SLURM_JOB_NAME} \
  --epochs 100 \
  --smac-initial-design DefaultConfiguration \
  --smac-pcs-file ~/prog/discoret/fair_hpo_smac/config/${SLURM_JOB_NAME}.pcs \
  --smac-runcount 1

exit 0
