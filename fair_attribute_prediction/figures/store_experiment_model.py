#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

if __name__ == "__main__":
    parser = ArgumentParser(fromfile_prefix_chars="+")
    parser.add_argument("--experiment")
    arguments = parser.parse_args()
    experiment_dir_path = Path(arguments.experiment)
    assert experiment_dir_path.is_dir()
    experiments_models_dir_path = Path("experiments") / "models"
    assert experiments_models_dir_path.is_dir()
    for state_file_name in ["parameters.pt", "final_model.pt", "best_model.pt"]:
        new_state_file_name = f"{experiment_dir_path.parts[-2]}-{state_file_name}"
        state_file_path = experiment_dir_path / state_file_name
        assert state_file_path.is_file()
        new_state_file_path = experiments_models_dir_path / new_state_file_name
        copy2(state_file_path, new_state_file_path)
        print(f"Copied {state_file_path} to {new_state_file_path}")
