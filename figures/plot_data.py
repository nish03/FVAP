#!/usr/bin/env python3

import json

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict


from torch import load, no_grad, tensor
from torch.backends import cudnn
from torch.linalg import vector_norm

from torchmetrics import JaccardIndex

from losses.fair_losses import fair_losses
from util import create_dataset, create_dataloader, create_model, get_device

parser = ArgumentParser(fromfile_prefix_chars="+")
parser.add_argument("--experiment_descriptions")


def evaluate_experiment(experiment_id: str, experiment_descriptions: Dict, uncalibrate: bool):
    experiments_models_dir_path = Path("experiments") / "models"
    parameter_file_path = experiments_models_dir_path / (experiment_id + "-parameters.pt")
    model_state_file_path = experiments_models_dir_path / (experiment_id + "-final_model.pt")

    parameters = load(parameter_file_path)
    parameters["pretrained_model"] = model_state_file_path

    train_dataset = create_dataset(parameters, split_name="train")
    eval_dataset = create_dataset(parameters, split_name=experiment_descriptions["split"])

    eval_dataloader = create_dataloader(parameters, eval_dataset)

    target_attribute = train_dataset.attribute(parameters["target_attribute_index"])
    target_attribute_values = []
    target_class_probabilities = []
    target_prediction_attribute_index = train_dataset.prediction_attribute_indices.index(target_attribute.index)

    sensitive_attribute = train_dataset.attribute(parameters["sensitive_attribute_index"])
    attribute_ious = [
        JaccardIndex(average=None, num_classes=target_attribute.size).to(get_device())
        for _ in range(sensitive_attribute.size)
    ]
    general_iou = JaccardIndex(num_classes=target_attribute.size).to(get_device())
    fair_loss_values = {fair_loss_name: 0.0 for fair_loss_name in fair_losses.keys()}
    sensitive_correct_prediction_counts = tensor([0] * sensitive_attribute.size)
    sensitive_attribute_image_counts = tensor([0] * sensitive_attribute.size)

    image_indices = []

    model = create_model(parameters, train_dataset)
    model.eval()

    with no_grad():
        for images, attributes, indices in eval_dataloader:
            images, attributes = images.to(get_device()), attributes.to(get_device())

            image_indices += indices.tolist()

            target_attribute.targets = attributes[:, target_attribute.index]
            sensitive_attribute.targets = attributes[:, sensitive_attribute.index]

            multi_output_class_logits = model(images)
            target_attribute.class_probabilities = model.module.attribute_class_probabilities(
                multi_output_class_logits, target_prediction_attribute_index
            )
            if uncalibrate:
                target_attribute.class_probabilities = target_attribute.class_probabilities / 4.0 * 3.0 + 0.125
                target_attribute.class_probabilities = (
                    target_attribute.class_probabilities - 0.5
                ) ** 5 / 0.5**4 + 0.5

            target_attribute.predictions = model.module.multi_attribute_predictions(multi_output_class_logits)[
                :, target_prediction_attribute_index
            ]
            is_correct_prediction = target_attribute.predictions.eq(target_attribute.targets)

            for sensitive_attribute_class_b in range(sensitive_attribute.size):
                from_class_b = sensitive_attribute.targets.eq(sensitive_attribute_class_b)
                if from_class_b.sum() == 0:
                    continue
                attribute_ious[sensitive_attribute_class_b](
                    target_attribute.predictions[from_class_b], target_attribute.targets[from_class_b]
                )

                sensitive_correct_prediction_counts[sensitive_attribute_class_b] += (
                    is_correct_prediction[from_class_b].sum().item()
                )
                sensitive_attribute_image_counts[sensitive_attribute_class_b] += from_class_b.sum().item()

            for fair_loss_name, fair_loss in fair_losses.items():
                fair_loss_values[fair_loss_name] += (
                    fair_loss(sensitive_attribute, target_attribute).item() * images.shape[0]
                )

            general_iou(target_attribute.predictions, target_attribute.targets)
            target_attribute_values += target_attribute.targets.tolist()
            target_class_probabilities += target_attribute.class_probabilities.tolist()

        image_count = sensitive_attribute_image_counts.sum().item()
        attribute_ious = [attribute_iou.compute().tolist() for attribute_iou in attribute_ious]
        attribute_iou_std = tensor(attribute_ious).mean(dim=1).std(dim=0).item()
        general_iou = general_iou.compute().item()

        for fair_loss_name, fair_loss in fair_losses.items():
            fair_loss_values[fair_loss_name] /= image_count

        sensitive_accuracies = (sensitive_correct_prediction_counts / sensitive_attribute_image_counts).tolist()
        accuracy = sensitive_correct_prediction_counts.sum().item() / image_count

    sorted_target_attribute_values = []
    sorted_target_class_probabilities = []
    for image_index, eval_dataset_index in sorted(zip(image_indices, range(len(image_indices)))):
        sorted_target_attribute_values.append(target_attribute_values[eval_dataset_index])
        sorted_target_class_probabilities.append(target_class_probabilities[eval_dataset_index])
    target_attribute_values = sorted_target_attribute_values
    target_class_probabilities = sorted_target_class_probabilities

    if experiment_id != baseline_experiment_id:
        experiment_descriptions["experiments"][experiment_id]["class_probability_changes"] = vector_norm(
            tensor(target_class_probabilities)
            - tensor(experiment_descriptions["experiments"][baseline_experiment_id]["target_class_probabilities"]),
            dim=1,
        ).tolist()

    if uncalibrate:
        new_experiment_id = experiment_id + "_uncalibrated"
        experiment_descriptions["experiments"][new_experiment_id] = {
            "loss": experiment_descriptions["experiments"][experiment_id]["loss"] + " (Uncalibrated)"
        }
        experiment_id = new_experiment_id

    print(f"experiment: {experiment_id}")

    experiment_descriptions["experiments"][experiment_id]["image_indices"] = image_indices
    experiment_descriptions["experiments"][experiment_id]["target_attribute_values"] = target_attribute_values
    experiment_descriptions["experiments"][experiment_id]["target_class_probabilities"] = target_class_probabilities
    experiment_descriptions["experiments"][experiment_id]["attribute_ious"] = attribute_ious
    print(f"  attribute_ious: {attribute_ious}")
    experiment_descriptions["experiments"][experiment_id]["general_iou"] = general_iou
    print(f"  general_iou: {general_iou}")
    experiment_descriptions["experiments"][experiment_id]["sensitive_accuracies"] = sensitive_accuracies
    print(f"  sensitive_attribute_accuracies: {sensitive_accuracies}")
    experiment_descriptions["experiments"][experiment_id]["accuracy"] = accuracy
    print(f"  accuracy: {accuracy}")
    for fair_loss_name, fair_loss_value in fair_loss_values.items():
        experiment_descriptions["experiments"][experiment_id][f"{fair_loss_name}_loss"] = fair_loss_value
        print(f"  {fair_loss_name}_loss: {fair_loss_value:.2E}")
    experiment_descriptions["experiments"][experiment_id]["attribute_iou_std"] = attribute_iou_std
    print(f"  attribute_iou_std: {attribute_iou_std:.3E}")
    experiment_descriptions["experiments"][experiment_id]["fair_loss_weight"] = parameters["fair_loss_weight"]
    print(f"  fair_loss_weight: {parameters['fair_loss_weight']:.2E}")


if __name__ == "__main__":
    arguments = parser.parse_args()
    with open(arguments.experiment_descriptions, "r") as experiment_descriptions_file:
        experiment_descriptions = json.load(experiment_descriptions_file)

    if cudnn.is_available():
        cudnn.enabled = False

    baseline_experiment_id = experiment_descriptions["baseline_experiment"]
    reference_experiment_ids = [
        experiment_id
        for experiment_id in experiment_descriptions["experiments"]
        if experiment_id != baseline_experiment_id
    ]

    evaluate_experiment(baseline_experiment_id, experiment_descriptions, uncalibrate=False)
    evaluate_experiment(baseline_experiment_id, experiment_descriptions, uncalibrate=True)
    for experiment_id in reference_experiment_ids:
        evaluate_experiment(experiment_id, experiment_descriptions, uncalibrate=False)

    with open(
        Path(arguments.experiment_descriptions).parent
        / (Path(arguments.experiment_descriptions).stem + "-processed.json"),
        "w",
    ) as experiment_descriptions_file:
        json.dump(experiment_descriptions, experiment_descriptions_file)
