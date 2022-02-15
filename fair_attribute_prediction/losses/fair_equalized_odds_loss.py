from typing import List

import torch
from torch import tensor

from multi_attribute_dataset import Attribute


def fair_equalized_odds_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_attribute_targets = multi_attribute_targets[:, sensitive_attribute.index]
    class_probabilities = model.module.attribute_class_probabilities(multi_output_class_logits, target_attribute.index)
    attribute_targets = multi_attribute_targets[:, target_attribute.index]
    ffp_differences = []
    for target_attribute_class in range(target_attribute.size):
        where_target_class = attribute_targets.eq(target_attribute_class)
        target_attribute_probabilities = class_probabilities[where_target_class]
        p_target_ground = target_attribute_probabilities.mean()  # p(y|y*)
        for sensitive_attribute_class in range(sensitive_attribute.size):
            where_sensitive_class = sensitive_attribute_targets.eq(sensitive_attribute_class)
            target_attribute_probabilities_sensitive_attribute = class_probabilities[
                where_target_class.logical_and(where_sensitive_class)
            ]
            p_target_ground_sensitive = target_attribute_probabilities_sensitive_attribute.mean()  # p(y|y*,s)
            ffp_differences.append((p_target_ground_sensitive - p_target_ground).pow(2))

    return tensor(ffp_differences).mean()
