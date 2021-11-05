#!/usr/bin/python

from pathlib import Path
from sys import argv
from os import chmod


for arg in argv[1:]:
    args_file_path = Path(arg)
    assert args_file_path.suffix == ".args"
    assert args_file_path.parts[0] == "config"
    job_name = "-".join(list(args_file_path.parts[1:-1]) + [args_file_path.stem])
    job_template = f"""\
#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --mem=100G
#SBATCH --chdir=/home/s9911486/prog/discoret/fair_hpo_smac/jobs
#SBATCH --job-name="{job_name}"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=tobias.haenel@tu-dresden.de
#SBATCH --mail-type=END,FAIL
#SBATCH --account=p_discoret

cd ~/prog/discoret/fair_hpo_smac

source jobs/load_alpha_modules.sh

export OMP_NUM_THREADS=${{SLURM_CPUS_ON_NODE}}

source env/bin/activate

./smac-hpo.py \\
  +{args_file_path} \\
  --in-memory-dataset

exit 0\
    """
    job_file_directory = Path("jobs") / "experiments" / Path(*args_file_path.parts[1:-1])
    job_file_directory.mkdir(parents=True, exist_ok=True)
    job_file_path = job_file_directory / f"{args_file_path.stem}.slurm"
    with open(job_file_path, "w") as job_file:
        print(job_template, file=job_file)
    job_file_path.chmod(0o755)
