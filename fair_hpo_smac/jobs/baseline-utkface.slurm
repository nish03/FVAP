#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="baseline-utkface"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=END,FAIL
#SBATCH --account=p_discoret

cd ~/prog/discoret/fair_hpo_smac

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

source env/bin/activate

./smac-hpo.py \
  --batch-size 144 \
  --datasplit-seed 42 \
  --epochs 200 \
  --image-size 64 \
  --in-memory-dataset \
  --output-dir experiments/baseline-utkface \
  --smac-initial-design DefaultConfiguration \
  --smac-pcs-file config/baseline-utkface.pcs \
  --smac-runcount 1 \
  --smac-seed 42 \
  --utkface-dir datasets/UTKFace

exit 0
