#!/usr/bin python

import json

from argparse import ArgumentParser
from pathlib import Path

from torch import load, no_grad, tensor
from torch.backends import cudnn

from util import create_dataset, create_dataloader, create_model, get_device

parser = ArgumentParser(fromfile_prefix_chars="+")
parser.add_argument("--experiment_descriptions")

if __name__ == "__main__":
    arguments = parser.parse_args()
    with open(arguments.experiment_descriptions, "r") as experiment_descriptions_file:
        experiment_descriptions = json.load(experiment_descriptions_file)

    if cudnn.is_available():
        cudnn.enabled = False

    for i, experiment_id in enumerate(experiment_descriptions["experiments"]):
        experiments_models_dir_path = Path("experiments") / "models"
        parameter_file_path = experiments_models_dir_path / (experiment_id + "-parameters.pt")
        model_state_file_path = experiments_models_dir_path / (experiment_id + "-final_model.pt")

        parameters = load(parameter_file_path)
        parameters["pretrained_model"] = model_state_file_path

        train_dataset = create_dataset(parameters, split_name="train")
        valid_dataset = create_dataset(parameters, split_name="valid")

        train_dataloader = create_dataloader(parameters, train_dataset)
        valid_dataloader = create_dataloader(parameters, valid_dataset)

        target_attribute = train_dataset.attribute(parameters["target_attribute_index"])
        target_attribute.targets = []
        target_attribute.class_probabilities = []
        target_prediction_attribute_index = train_dataset.prediction_attribute_indices.index(target_attribute.index)

        model = create_model(parameters, train_dataset)
        model.eval()

        with no_grad():
            for images, attributes in valid_dataloader:
                images, attributes = images.to(get_device()), attributes.to(get_device())

                multi_output_class_logits = model(images)
                target_attribute.targets += attributes[:, target_attribute.index].tolist()
                target_attribute.class_probabilities += model.module.attribute_class_probabilities(
                    multi_output_class_logits, target_prediction_attribute_index
                ).tolist()

        experiment_descriptions["experiments"][experiment_id]["targets"] = target_attribute.targets
        experiment_descriptions["experiments"][experiment_id][
            "class_probabilities"
        ] = target_attribute.class_probabilities
        target_attribute.targets = tensor(target_attribute.targets)
        target_attribute.class_probabilities = tensor(target_attribute.class_probabilities)

    with open(
        Path(arguments.experiment_descriptions).parent
        / (Path(arguments.experiment_descriptions).stem + "-processed.json"),
        "w",
    ) as experiment_descriptions_file:
        json.dump(experiment_descriptions, experiment_descriptions_file)
