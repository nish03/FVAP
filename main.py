#!/usr/bin/env python3
from glob import glob
from pathlib import Path
import sys

from experiment import run_experiment

if __name__ == "__main__":
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
