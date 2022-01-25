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
    target_predictions = model.module.multi_attribute_predictions(multi_output_class_logits)[:, target_attribute.index]
    demographic_loss = 0
    for target_attribute_class in range(target_attribute.size):
        where_target_class = target_predictions.eq(target_attribute_class)
        frequency_target_class = where_target_class.sum()
        p_target_class = frequency_target_class/target_predictions.size(0)
        for sensitive_attribute_class in range(sensitive_attribute.size):
            where_sensitive_class = sensitive_attribute_targets.eq(sensitive_attribute_class)
            frequency_sensitive_class = where_sensitive_class.sum()
            index = where_target_class & where_sensitive_class
            frequency_target_class_sensitive_class = index.sum()
            try:
                p_conditioned = frequency_target_class_sensitive_class/frequency_sensitive_class
                demographic_loss = demographic_loss + (p_conditioned - p_target_class).pow(2)
            except:
                pass
    return tensor(demographic_loss)
