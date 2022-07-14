from typing import List

import torch
from torch import nonzero, tensor
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    nonzero_probabilities = probabilities[nonzero(probabilities, as_tuple=True)]
    return -(nonzero_probabilities * nonzero_probabilities.log()).sum()


def fair_mutual_information_dp_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_class_probabilities = one_hot(sensitive_attribute.targets, num_classes=sensitive_attribute.size).float()
    joint_class_probabilities = sensitive_class_probabilities.unsqueeze(
        dim=2
    ) * target_attribute.class_probabilities.unsqueeze(dim=1)
    p_sensitive = sensitive_class_probabilities.mean(dim=0)
    p_pred = target_attribute.class_probabilities.mean(dim=0)
    p_pred_sensitive = joint_class_probabilities.mean(dim=0)
    entropy_sensitive = entropy(p_sensitive)
    entropy_target = entropy(p_pred)
    joint_entropy = entropy(p_pred_sensitive)
    mi = entropy_sensitive + entropy_target - joint_entropy
    return mi


def fair_mutual_information_eo_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_class_probabilities = one_hot(sensitive_attribute.targets, num_classes=sensitive_attribute.size).float()
    joint_class_probabilities = sensitive_class_probabilities.unsqueeze(
        dim=2
    ) * target_attribute.class_probabilities.unsqueeze(dim=1)
    mi = 0.0
    for target_class_a in range(target_attribute.size):
        from_target_class_a = target_attribute.targets.eq(target_class_a)
        if from_target_class_a.sum() == 0:
            return tensor(0.0, device=sensitive_attribute.targets.device)

        p_sensitive = sensitive_class_probabilities[from_target_class_a].mean(dim=0)
        p_pred = target_attribute.class_probabilities[from_target_class_a].mean(dim=0)
        p_pred_sensitive_target = joint_class_probabilities[from_target_class_a].mean(dim=0)
        entropy_sensitive = entropy(p_sensitive)
        entropy_target = entropy(p_pred)
        joint_entropy = entropy(p_pred_sensitive_target)
        mi = mi + entropy_sensitive + entropy_target - joint_entropy
    return mi
