#!/usr/bin/env python3
from glob import glob

from pathlib import Path
import sys

from experiment import run_experiment

if __name__ == "__main__":
    """
    Runs fair attribute prediction experiments from the command line. The hyper parameters have to be specified in
    argument files.
    
    Execution:
        ./main.py ARGS_ROOT_DIR [REL_ARGS_FILE...]
    
    Arguments:
        ARGS_ROOT_DIR - Path to the arguments root directory for the set of experiments
        REL_ARGS_FILE - (Optional) Path to one or more the arguments files relative to the arguments root directory
                        All argument files inside ARGS_ROOT_DIR are used, if unspecified
    
    Argument file syntax:
        See documentation of run_experiment() (experiment.py)
    """
    relative_args_file_paths = []
    if len(sys.argv) < 2:
        raise ValueError(f"Root argument directory wasn't specified")

    args_root_dir_path = Path(sys.argv[1])
    if not args_root_dir_path.is_dir():
        raise ValueError(f"{args_root_dir_path} is not a directory")

    args_files = []
    if len(sys.argv) == 2:
        args_file_path_strs = glob(str(args_root_dir_path / "**" / "*.args"), recursive=True)
        relative_args_file_paths = [Path(args_file_path_str) for args_file_path_str in args_file_path_strs]
    else:
        relative_args_file_paths = [Path(relative_args_file_path_str) for relative_args_file_path_str in sys.argv[2:]]

    relative_args_file_paths.sort()

    for relative_args_file_path in relative_args_file_paths:
        run_experiment(args_root_dir_path, relative_args_file_path)
