from itertools import combinations
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
    iou = 0
    for attribute_class in range(attribute_size):
        iou = iou + confusion_matrix[attribute_class, attribute_class] / (
            confusion_matrix[attribute_class, :].sum()
            + confusion_matrix[:, attribute_class].sum()
            - confusion_matrix[attribute_class, attribute_class]
        )
    iou /= attribute_size
    return iou


def sensitive_intersection_over_union(
    from_sensitive_class: torch.Tensor,
    target_class_probabilities: torch.Tensor,
    target_attribute_targets: torch.Tensor,
) -> torch.Tensor:
    class_probabilities = target_class_probabilities[from_sensitive_class]
    attribute_targets = target_attribute_targets[from_sensitive_class]
    sensitive_iou = intersection_over_union(class_probabilities, attribute_targets)
    return sensitive_iou


def fair_intersection_over_union_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    fair_iou_loss = 0.0
    for sensitive_class_a, sensitive_class_b in combinations(range(sensitive_attribute.size), 2):
        from_sensitive_class_a = sensitive_attribute.targets.eq(sensitive_class_a)
        from_sensitive_class_b = sensitive_attribute.targets.eq(sensitive_class_b)
        if from_sensitive_class_a.sum() == 0 or from_sensitive_class_b.sum() == 0:
            print(f"no samples for sensitive class combination ({sensitive_class_a}, {sensitive_class_b})")
            return tensor(0.0, device=sensitive_attribute.targets.device)
        iou_a = sensitive_intersection_over_union(
            from_sensitive_class_a, target_attribute.class_probabilities, target_attribute.targets
        )
        iou_b = sensitive_intersection_over_union(
            from_sensitive_class_b, target_attribute.class_probabilities, target_attribute.targets
        )
        fair_iou_loss = fair_iou_loss + (iou_a - iou_b).pow(2)

    fair_iou_loss /= sensitive_attribute.size * (sensitive_attribute.size - 1) / 2
    return fair_iou_loss
