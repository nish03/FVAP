#!/usr/bin python

import json
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

from matplotlib import pyplot as plt
from torch import tensor, load

from util import create_dataset

parser = ArgumentParser(fromfile_prefix_chars="+")
parser.add_argument("--experiment_descriptions")

if __name__ == "__main__":
    arguments = parser.parse_args()
    experiment_descriptions_file_path = Path(arguments.experiment_descriptions)
    with open(experiment_descriptions_file_path, "r") as experiment_descriptions_file:
        experiment_descriptions = json.load(experiment_descriptions_file)

    figures_data_dir = Path("figures") / "data"

    significant_image_count = 10

    baseline_experiment_id = experiment_descriptions["baseline_experiment"]

    for experiment_id in experiment_descriptions["experiments"]:
        if experiment_id.startswith(baseline_experiment_id):
            continue

        sorted_class_probability_changes, image_indices = tensor(
            experiment_descriptions["experiments"][experiment_id]["class_probability_changes"]
        ).sort()

        experiments_models_dir_path = Path("experiments") / "models"
        parameter_file_path = experiments_models_dir_path / (experiment_id + "-parameters.pt")
        parameters = load(parameter_file_path)
        eval_dataset = create_dataset(parameters, split_name=experiment_descriptions["split"])

        experiment_data_dir = figures_data_dir / "significant_images" / experiment_id
        experiment_data_dir.mkdir(parents=True, exist_ok=True)
        significant_image_changes = {
            "image_indices": [],
            "target_attribute_values": [],
            "baseline_target_class_probabilities": [],
            "reference_target_class_probabilities": [],
        }
        for k in range(1, significant_image_count + 1):
            significant_image_index = image_indices[-k]
            image, attribute_values, _ = eval_dataset[significant_image_index]
            plt.imshow(image.permute(1, 2, 0))
            significant_image_file_path = experiment_data_dir / f"image_{k}.png"
            plt.savefig(significant_image_file_path)
            significant_image_changes["target_attribute_values"].append(
                experiment_descriptions["experiments"][experiment_id]["target_attribute_values"][
                    significant_image_index
                ]
            )
            significant_image_changes["reference_target_class_probabilities"].append(
                experiment_descriptions["experiments"][experiment_id]["target_class_probabilities"][
                    significant_image_index
                ]
            )
            significant_image_changes["baseline_target_class_probabilities"].append(
                experiment_descriptions["experiments"][baseline_experiment_id]["target_class_probabilities"][
                    significant_image_index
                ]
            )
        significant_image_changes_file_path = experiment_data_dir / "significant_image_changes.json"
        with open(significant_image_changes_file_path, "w") as significant_image_changes_file:
            json.dump(significant_image_changes, significant_image_changes_file)
