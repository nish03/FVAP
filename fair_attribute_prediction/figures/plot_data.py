# !/usr/bin python

import json

from argparse import ArgumentParser
from pathlib import Path

from torch import load, no_grad, tensor, zeros
from torch.backends import cudnn
from torch.nn.functional import one_hot

from losses.fair_intersection_over_union_loss import intersection_over_union
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
        target_attribute_values = []
        target_attribute_class_probabilities = []
        target_prediction_attribute_index = train_dataset.prediction_attribute_indices.index(target_attribute.index)

        sensitive_attribute = train_dataset.attribute(parameters["sensitive_attribute_index"])
        iou_attribute_losses = [[0.0] * target_attribute.size] * sensitive_attribute.size
        iou_general_loss = 0.0

        model = create_model(parameters, train_dataset)
        model.eval()

        with no_grad():
            for images, attributes in valid_dataloader:
                images, attributes = images.to(get_device()), attributes.to(get_device())

                batch_multi_output_class_logits = model(images)
                batch_target_class_probabilities = model.module.attribute_class_probabilities(
                    batch_multi_output_class_logits, target_prediction_attribute_index
                )
                batch_target_attribute_values = attributes[:, target_attribute.index]
                batch_sensitive_attribute_values = attributes[:, sensitive_attribute.index]

                for sensitive_attribute_class_b in range(sensitive_attribute.size):
                    from_class_b = batch_sensitive_attribute_values.eq(sensitive_attribute_class_b)
                    if from_class_b.sum() == 0:
                        continue
                    confusion_matrix = (
                        batch_target_class_probabilities[from_class_b, :].unsqueeze(dim=2)
                        * one_hot(batch_target_attribute_values[from_class_b], target_attribute.size).unsqueeze(dim=1)
                    ).sum(dim=0)
                    for target_attribute_class_a in range(target_attribute.size):
                        iou_attribute_losses[sensitive_attribute_class_b][target_attribute_class_a] += confusion_matrix[
                            target_attribute_class_a, target_attribute_class_a
                        ] / (
                            confusion_matrix[target_attribute_class_a, :].sum()
                            + confusion_matrix[:, target_attribute_class_a].sum()
                            - confusion_matrix[target_attribute_class_a, target_attribute_class_a]
                        )

                iou_general_loss += intersection_over_union(
                    batch_target_class_probabilities, batch_target_attribute_values
                ).item()
                target_attribute_values.append(batch_target_attribute_values.tolist())
                target_attribute_class_probabilities.append(batch_target_class_probabilities.tolist())

            iou_attribute_losses = (
                tensor(iou_attribute_losses) / (sensitive_attribute.size * target_attribute.size)
            ).tolist()
            iou_general_loss /= sensitive_attribute.size

        experiment_descriptions["experiments"][experiment_id]["target_attribute_values"] = target_attribute_values
        experiment_descriptions["experiments"][experiment_id][
            "target_class_probabilities"
        ] = target_attribute_class_probabilities
        experiment_descriptions["experiments"][experiment_id]["iou_attribute_losses"] = iou_attribute_losses
        experiment_descriptions["experiments"][experiment_id]["iou_general_loss"] = iou_general_loss

    with open(
        Path(arguments.experiment_descriptions).parent
        / (Path(arguments.experiment_descriptions).stem + "-processed.json"),
        "w",
    ) as experiment_descriptions_file:
        json.dump(experiment_descriptions, experiment_descriptions_file)
