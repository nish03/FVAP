from typing import List

import torch
from torch import tensor

from multi_attribute_dataset import Attribute


def fair_demographic_parity_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_attribute_targets = multi_attribute_targets[:, sensitive_attribute.index]
    class_probabilities = model.module.attribute_class_probabilities(multi_output_class_logits, target_attribute.index)
    demographic_loss = 0
    for target_attribute_class in range(target_attribute.size):
        target_attribute_probabilities = class_probabilities[:, target_attribute_class]
        p_target_attribute = target_attribute_probabilities.mean()
        for sensitive_attribute_class in range(sensitive_attribute.size):
            where_sensitive_class = sensitive_attribute_targets.eq(sensitive_attribute_class)
            target_attribute_probabilities_sensitive_attribute = target_attribute_probabilities[where_sensitive_class]
            p_conditioned = target_attribute_probabilities_sensitive_attribute.mean()
            demographic_loss = demographic_loss + (p_conditioned - p_target_attribute).pow(2)
    return demographic_loss
