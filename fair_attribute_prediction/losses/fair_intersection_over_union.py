from typing import List

import torch
from torch import tensor
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def intersection_over_union(
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


def sensitive_intersection_over_union(
    from_sensitive_class: torch.Tensor,
    target_class_probabilities: torch.Tensor,
    target_attribute_targets: torch.Tensor,
) -> torch.Tensor:
    class_probabilities = target_class_probabilities[from_sensitive_class]
    attribute_targets = target_attribute_targets[from_sensitive_class]
    _sensitive_iou = intersection_over_union(class_probabilities, attribute_targets)
    return _sensitive_iou


def fair_intersection_over_union_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_attribute_targets = multi_attribute_targets[:, sensitive_attribute.index]
    target_class_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    target_attribute_targets = multi_attribute_targets[:, target_attribute.index]
    squared_iou_differences = []
    for sensitive_class_a in range(sensitive_attribute.size):
        from_sensitive_class_a = sensitive_attribute_targets.eq(sensitive_class_a)
        if from_sensitive_class_a.sum() == 0:
            return tensor(0.0, device=sensitive_attribute_targets.device)
        sensitive_iou_a = sensitive_intersection_over_union(
            from_sensitive_class_a, target_class_probabilities, target_attribute_targets
        )
        for sensitive_class_b in range(sensitive_class_a):
            from_sensitive_class_b = sensitive_attribute_targets.eq(sensitive_class_b)
            if from_sensitive_class_b.sum() == 0:
                return tensor(0.0, device=sensitive_attribute_targets.device)
            sensitive_iou_b = sensitive_intersection_over_union(
                from_sensitive_class_b, target_class_probabilities, target_attribute_targets
            )

            squared_iou_difference = (sensitive_iou_a - sensitive_iou_b).pow(2)
            squared_iou_differences.append(squared_iou_difference)

    _fair_iou_loss = tensor(squared_iou_differences).mean()
    return _fair_iou_loss
