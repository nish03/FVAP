#!/bin/bash

shopt -s nullglob

sbatch <<EOT
#!/bin/bash -l
 
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=alpha
#SBATCH --account=p_da_studenten
#SBATCH --mem-per-cpu=10G
#SBATCH --output=slurm-%x-%A-%a.out
#SBATCH --error=slurm-%x-%A-%a.err
#SBATCH --job-name="fair_attribute_prediction-$(date -Iseconds)"

source ~/prog/discoret/load_alpha_modules.sh

export OMP_NUM_THREADS=\$SLURM_CPUS_ON_NODE
unset XDG_RUNTIME_DIR

cd ~/prog/discoret/fairclassifier/fair_attribute_prediction
 
srun ./main.py ${@}
EOT
