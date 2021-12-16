from typing import Dict, List, Optional, Tuple
from statistics import mean

import comet_ml
import torch.utils.data
from torch import tensor
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot

from Metrics import MetricsState, metrics
from MultiAttributeDataset import Attribute
from Util import get_device


def fair_mi_criterion(
    model: torch.nn.Module,
    multi_output_class_logits: torch.Tensor,
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    return tensor(0.0, device=get_device())


def intersection_over_union(
    sensitive_class: int,
    sensitive_attribute_targets: torch.Tensor,
    target_class_probabilities: torch.Tensor,
    target_attribute_targets: torch.Tensor,
):
    is_sensitive_class = sensitive_attribute_targets == sensitive_class
    class_probabilities = target_class_probabilities[is_sensitive_class]
    attribute_targets = target_attribute_targets[is_sensitive_class]
    target_attribute_size = class_probabilities.shape[1]
    confusion_matrix = (
        class_probabilities.unsqueeze(dim=2) * one_hot(attribute_targets, target_attribute_size).unsqueeze(dim=1)
    ).sum(dim=0)
    target_class_ious = []
    for target_class in range(target_attribute_size):
        target_class_iou = confusion_matrix[target_class, target_class] / (
            confusion_matrix[target_class, :].sum()
            + confusion_matrix[:, target_class].sum()
            - confusion_matrix[target_class, target_class]
        )
        target_class_ious.append(target_class_iou)
    iou = mean(target_class_ious)
    return iou


def fair_intersection_over_union(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_attribute_targets = multi_attribute_targets[:, sensitive_attribute.index]
    target_class_probabilities = model.module.attribute_class_probabilities(multi_output_class_logits, target_attribute)
    target_attribute_targets = multi_attribute_targets[:, target_attribute.index]
    sensitive_iou_squared_differences = []
    for sensitive_class_1 in range(sensitive_attribute.size):
        for sensitive_class_2 in range(sensitive_class_1):
            sensitive_iou_1 = intersection_over_union(
                sensitive_class_1, sensitive_attribute_targets, target_class_probabilities, target_attribute_targets
            )
            sensitive_iou_2 = intersection_over_union(
                sensitive_class_2,
                sensitive_attribute_targets,
                target_class_probabilities,
                target_attribute_targets,
            )
            sensitive_iou_squared_difference = (sensitive_iou_1 - sensitive_iou_2).pow()
            sensitive_iou_squared_differences.append(sensitive_iou_squared_difference)

    fair_iou_loss = mean(sensitive_iou_squared_differences)
    return fair_iou_loss


def cross_entropy_criterion(
    model: torch.nn.Module, multi_output_class_logits: List[torch.Tensor], multi_attribute_targets: torch.Tensor
) -> torch.Tensor:
    output_losses = []
    for attribute_size, output_class_logits in zip(model.module.unique_attribute_sizes, multi_output_class_logits):
        output_attribute_targets = multi_attribute_targets[:, model.module.attribute_sizes.eq(attribute_size)]
        if attribute_size == 2:
            output_attribute_targets = output_attribute_targets.float()
            output_loss = binary_cross_entropy_with_logits(
                output_class_logits, output_attribute_targets, reduction="sum"
            )
        else:
            output_loss = cross_entropy(output_class_logits, output_attribute_targets, reduction="sum")
        output_losses.append(output_loss)

    cross_entropy_loss = sum(output_losses) / multi_attribute_targets.numel()
    return cross_entropy_loss


fair_criterions = {"iou": fair_intersection_over_union, "mi": fair_mi_criterion}


def criterion(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    parameters: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cross_entropy_loss = cross_entropy_criterion(model, multi_output_class_logits, multi_attribute_targets)

    sensitive_attribute: Attribute = parameters["sensitive_attribute"]
    target_attribute: Attribute = parameters["target_attribute"]
    fair_loss_type: str = parameters["fair_loss_type"]
    fair_loss_weight: float = parameters["fair_loss_weight"]
    fair_criterion = fair_criterions[fair_loss_type]
    fair_loss = fair_criterion(
        model, multi_output_class_logits, multi_attribute_targets, sensitive_attribute, target_attribute
    )

    loss = cross_entropy_loss + fair_loss_weight * fair_loss
    loss_term_values = {"cross_entropy": cross_entropy_loss.item(), f"fair_{fair_loss_type}": fair_loss.item()}

    return loss, loss_term_values


def loss_with_metrics(
    model: torch.nn.Module,
    batch_data: (torch.Tensor, torch.Tensor),
    metrics_state: Optional[MetricsState],
    parameters: Dict,
) -> Tuple[torch.Tensor, Dict[str, float], MetricsState]:
    images, multi_attribute_targets = batch_data[0].to(get_device()), batch_data[1].to(get_device())

    multi_output_class_logits = model(images)

    loss, loss_term_values = criterion(model, multi_output_class_logits, multi_attribute_targets, parameters)

    loss_value = loss.item()
    _metrics, metrics_state = metrics(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        loss_value,
        loss_term_values,
        metrics_state,
    )
    return loss, _metrics, metrics_state


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_count: int,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    parameters: Dict,
    experiment: comet_ml.Experiment,
) -> Tuple[Dict, Dict]:
    best_model_state = {}
    best_valid_loss = None
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_evaluation_state = None
        with experiment.context_manager("train"):
            for batch_data in train_dataloader:
                optimizer.zero_grad(set_to_none=True)

                loss, train_metrics, train_evaluation_state = loss_with_metrics(
                    model, batch_data, train_evaluation_state, parameters
                )

                loss.backward()
                optimizer.step()

            experiment.log_metrics(train_metrics, epoch=epoch)

        model.eval()
        valid_evaluation_state = None
        with experiment.context_manager("valid"):
            for batch_data in valid_dataloader:
                loss, valid_metrics, valid_evaluation_state = loss_with_metrics(
                    model, batch_data, valid_evaluation_state, parameters
                )
            experiment.log_metrics(valid_metrics, epoch=epoch)
        epoch_valid_loss = valid_metrics["loss"]
        if best_valid_loss is None or best_valid_loss < epoch_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_state = {
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

    final_model_state = {
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
