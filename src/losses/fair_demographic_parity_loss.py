from typing import List

import torch
from torch import tensor

from multi_attribute_dataset import Attribute


def fair_demographic_parity_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    """
    Computes the fair l2-demographic parity loss from sensitive attribute labels and target attribute class probabilities.

    :param sensitive_attribute: Sensitive Attribute
        with targets member (Tensor[sample_count]) containing class labels of each sample
    :param target_attribute: Target Attribute
        with class_probabilities member (Tensor[sample_count, class_count]) containing predicted probabilities of each
        sample and class
    :return: Tensor[] containing the differentiable loss value
    """
    demographic_loss = 0
    for target_class_a in range(target_attribute.size):
        target_class_a_probabilities = target_attribute.class_probabilities[:, target_class_a]
        p_pred = target_class_a_probabilities.mean()  # p(y=a)
        for sensitive_class_b in range(sensitive_attribute.size):
            from_sensitive_class_b = sensitive_attribute.targets.eq(sensitive_class_b)
            if from_sensitive_class_b.sum() == 0:
                return tensor(0, device=sensitive_attribute.targets.device)
            target_class_a_sensitive_class_b_probabilities = target_class_a_probabilities[
                from_sensitive_class_b
            ]  # p(y=a|s=b)
            p_pred_sensitive = target_class_a_sensitive_class_b_probabilities.mean()
            demographic_loss = demographic_loss + (p_pred_sensitive - p_pred).pow(2)
    return demographic_loss
