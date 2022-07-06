from typing import List

import torch
from torch import tensor

from multi_attribute_dataset import Attribute


def fair_equalized_odds_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    equalized_odds = 0
    for target_class_a in range(target_attribute.size):
        target_class_a_probabilities = target_attribute.class_probabilities[:, target_class_a]  # p(y=a)
        for target_class_b in range(target_attribute.size):
            from_target_class_b = target_attribute.targets.eq(target_class_b)
            if from_target_class_b.sum() == 0:
                return tensor(0.0, device=target_attribute.targets.device)
            target_class_a_class_b_probabilities = target_class_a_probabilities[from_target_class_b]
            p_target_ground = target_class_a_class_b_probabilities.mean()  # p(y=a|y*=b)
            for sensitive_class_c in range(sensitive_attribute.size):
                from_sensitive_class_c = sensitive_attribute.targets.eq(sensitive_class_c)
                from_target_class_b_and_sensitive_class_c = from_target_class_b.logical_and(from_sensitive_class_c)
                if from_target_class_b_and_sensitive_class_c.sum() == 0:
                    return tensor(0.0, device=target_attribute.targets.device)
                target_class_a_class_b_sensitive_class_c_probabilities = target_class_a_probabilities[
                    from_target_class_b_and_sensitive_class_c
                ]
                p_target_ground_sensitive = (
                    target_class_a_class_b_sensitive_class_c_probabilities.mean()
                )  # p(y|y*=b, s=c)
                equalized_odds = equalized_odds + (p_target_ground_sensitive - p_target_ground).pow(2)

    return equalized_odds
