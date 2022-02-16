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
    equalized_odds = 0
    for target_class_a in range(target_attribute.size):
        target_class_a_probabilities = class_probabilities[:, target_class_a]  # p(y=a)
        for target_class_b in range(target_attribute.size):
            from_target_class_b = attribute_targets.eq(target_class_b)
            if from_target_class_b.sum() == 0:
                print(f"no samples from target class b ({target_class_b})")
                return tensor(0.0, device=attribute_targets.device)
            target_class_a_class_b_probabilities = target_class_a_probabilities[from_target_class_b]
            p_target_ground = target_class_a_class_b_probabilities.mean()  # p(y=a|y*=b)
            for sensitive_class_c in range(sensitive_attribute.size):
                from_sensitive_class_c = sensitive_attribute_targets.eq(sensitive_class_c)
                from_target_class_b_and_sensitive_class_c = from_target_class_b.logical_and(from_sensitive_class_c)
                if from_target_class_b_and_sensitive_class_c.sum() == 0:
                    print(
                        f"no samples from target class b ({target_class_b}) and sensitive class c ({sensitive_class_c})"
                    )
                    return tensor(0.0, device=attribute_targets.device)
                target_class_a_class_b_sensitive_class_c_probabilities = target_class_a_probabilities[
                    from_target_class_b_and_sensitive_class_c
                ]
                p_target_ground_sensitive = (
                    target_class_a_class_b_sensitive_class_c_probabilities.mean()
                )  # p(y|y*=b, s=c)
                equalized_odds = equalized_odds + (p_target_ground_sensitive - p_target_ground).pow(2)

    return equalized_odds
