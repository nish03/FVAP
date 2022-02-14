#!/usr/bin/env python3
from glob import glob
from pathlib import Path
import sys

from experiment import run_experiment
from tqdm import tqdm

if __name__ == "__main__":
    args_root_dir_paths = []
    for args_root_dir in sys.argv[1:]:
        args_root_dir_path = Path(args_root_dir)
        if not args_root_dir_path.is_dir():
            raise ValueError(f"{args_root_dir_path} is not a directory")
        args_root_dir_paths.append(args_root_dir_path)

    args_file_paths = []
    for args_root_dir_path in args_root_dir_paths:
        args_files = glob(str(args_root_dir_path / "**" / "*.args"), recursive=True)
        relative_args_file_paths = []
        for args_file in args_files:
            relative_args_file_path = Path(args_file).relative_to(args_root_dir_path)
            args_file_paths.append((args_root_dir_path, relative_args_file_path))

    args_file_paths.sort()

    for args_root_dir_path, relative_args_file_path in tqdm(args_file_paths):
        run_experiment(args_root_dir_path, relative_args_file_path)
