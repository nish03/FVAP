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
    target_class_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    demographic_loss = 0
    for target_class_a in range(target_attribute.size):
        target_class_a_probabilities = target_class_probabilities[:, target_class_a]
        p_pred = target_class_a_probabilities.mean()  # p(y=a)
        for sensitive_class_b in range(sensitive_attribute.size):
            from_sensitive_class_b = sensitive_attribute_targets.eq(sensitive_class_b)
            if from_sensitive_class_b.sum() == 0:
                print(f"no samples from sensitive class b ({sensitive_class_b})")
                return tensor(0, device=multi_attribute_targets.device)
            target_class_a_sensitive_class_b_probabilities = target_class_a_probabilities[
                from_sensitive_class_b
            ]  # p(y=a|s=b)
            p_pred_sensitive = target_class_a_sensitive_class_b_probabilities.mean()
            demographic_loss = demographic_loss + (p_pred_sensitive - p_pred).pow(2)
    return demographic_loss
