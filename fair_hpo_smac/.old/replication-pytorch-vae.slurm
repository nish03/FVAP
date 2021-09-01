#!/bin/bash


#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu2
#SBATCH --mem-per-cpu=2583M
#SBATCH -J "replication-pytorch-vae"
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_discoret

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

source ~/.virtualenvs/discoret/bin/activate
cd ~/workspaces/ssd/s9911486-discoret
~/prog/discoret/fair_hpo_smac/replication-pytorch-vae.py \
  --celeba-dir datasets/CelebA \
  --output-dir experiments/replication-pytorch-vae

exit 0
