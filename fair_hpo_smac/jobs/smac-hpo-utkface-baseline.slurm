#!/bin/bash


#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="smac-hpo-utkface-baseline"
#SBATCH --output=slurm-%x-%j.out
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

source ~/workspaces/ssd/s9911486-virtualenvs/discoret/bin/activate

cd ~/workspaces/ssd/s9911486-discoret
~/prog/discoret/fair_hpo_smac/smac-hpo.py \
  --utkface-dir datasets/UTKFace \
  --output-dir experiments/${SLURM_JOB_NAME} \
  --cost MS-SSIM \
  --smac-pcs-file ~/prog/discoret/fair_hpo_smac/config/${SLURM_JOB_NAME}.pcs \
  --epochs 100 \
  --smac-runtime 84600

exit 0
