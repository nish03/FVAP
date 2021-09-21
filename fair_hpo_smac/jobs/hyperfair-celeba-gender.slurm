#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="hyperfair-celeba-gender"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=END,FAIL
#SBATCH --account=p_discoret

cd ~/prog/discoret/fair_hpo_smac

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

source env/bin/activate

./smac-hpo.py \
  --batch-size 512 \
  --celeba-dir datasets/CelebA \
  --cost FairMS-SSIM \
  --epochs 50 \
  --image-size 64 \
  --output-dir experiments/hyperfair-celeba-gender \
  --sensitive-attribute 1 \
  --smac-initial-design Sobol \
  --smac-pcs-file config/hyperfair-celeba-gender.pcs \
  --smac-runtime 84600 \
  --smac-seed 42 \
  --in-memory-dataset

exit 0
