from math import nan
from typing import Dict, List, Optional, Tuple
import comet_ml
import torch.utils.data
from torch import tensor
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot

from Metrics import MetricsState, metrics
from MultiAttributeDataset import Attribute
from Util import get_device


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.log()).sum()


def fair_mi_criterion(
    model: torch.nn.Module,
    multi_output_class_logits: torch.Tensor,
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    batch_target_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    batch_sensitive_probabilities = one_hot(multi_attribute_targets[:, sensitive_attribute.index]).float()
    batch_joint_probabilities = batch_sensitive_probabilities.unsqueeze(dim=2) * batch_target_probabilities.unsqueeze(
        dim=1
    )
    sensitive_probabilities = batch_sensitive_probabilities.mean(dim=0)
    target_probabilities = batch_target_probabilities.mean(dim=0)
    joint_probabilities = batch_joint_probabilities.mean(dim=0)
    sensitive_entropy = entropy(sensitive_probabilities)
    target_entropy = entropy(target_probabilities)
    joint_entropy = entropy(joint_probabilities)
    mi = sensitive_entropy + target_entropy - joint_entropy
    return mi


def iou(
    class_probabilities: torch.Tensor,
    attribute_targets: torch.Tensor,
) -> torch.Tensor:
    attribute_size = class_probabilities.shape[1]
    confusion_matrix = (
        class_probabilities.unsqueeze(dim=2) * one_hot(attribute_targets, attribute_size).unsqueeze(dim=1)
    ).sum(dim=0)
    class_ious = []
    for attribute_class in range(attribute_size):
        class_iou = confusion_matrix[attribute_class, attribute_class] / (
            confusion_matrix[attribute_class, :].sum()
            + confusion_matrix[:, attribute_class].sum()
            - confusion_matrix[attribute_class, attribute_class]
        )
        class_ious.append(class_iou)
    _iou = tensor(class_ious).mean()
    return _iou


def sensitive_iou(
    from_sensitive_class: torch.Tensor,
    target_class_probabilities: torch.Tensor,
    target_attribute_targets: torch.Tensor,
) -> torch.Tensor:
    class_probabilities = target_class_probabilities[from_sensitive_class]
    attribute_targets = target_attribute_targets[from_sensitive_class]
    _sensitive_iou = iou(class_probabilities, attribute_targets)
    return _sensitive_iou


def fair_iou_criterion(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:

    sensitive_attribute_targets = multi_attribute_targets[:, sensitive_attribute.index]
    class_probabilities = model.module.attribute_class_probabilities(multi_output_class_logits, target_attribute.index)
    attribute_targets = multi_attribute_targets[:, target_attribute.index]
    squared_iou_differences = []
    for sensitive_class_a in range(sensitive_attribute.size):
        from_sensitive_class_a = sensitive_attribute_targets.eq(sensitive_class_a)
        if from_sensitive_class_a.sum() == 0:
            return tensor(0.0, device=sensitive_attribute_targets.device)
        sensitive_iou_a = sensitive_iou(from_sensitive_class_a, class_probabilities, attribute_targets)
        for sensitive_class_b in range(sensitive_class_a):
            from_sensitive_class_b = sensitive_attribute_targets.eq(sensitive_class_b)
            if from_sensitive_class_b.sum() == 0:
                return tensor(0.0, device=sensitive_attribute_targets.device)
            sensitive_iou_b = sensitive_iou(from_sensitive_class_b, class_probabilities, attribute_targets)

            squared_iou_difference = (sensitive_iou_a - sensitive_iou_b).pow(2)
            squared_iou_differences.append(squared_iou_difference)

    fair_iou_loss = tensor(squared_iou_differences).mean()
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


fair_criterions = {"iou": fair_iou_criterion, "mi": fair_mi_criterion}


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
