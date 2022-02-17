from typing import List

import torch
from torch import nonzero, tensor
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    nonzero_probabilities = probabilities[nonzero(probabilities, as_tuple=True)]
    return -(nonzero_probabilities * nonzero_probabilities.log()).sum()


def fair_mutual_information_dp_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    pred_target_class_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    sensitive_class_probabilities = one_hot(
        multi_attribute_targets[:, sensitive_attribute.index], num_classes=sensitive_attribute.size
    ).float()
    joint_class_probabilities = sensitive_class_probabilities.unsqueeze(
        dim=2
    ) * pred_target_class_probabilities.unsqueeze(dim=1)
    p_sensitive = sensitive_class_probabilities.mean(dim=0)
    p_pred = pred_target_class_probabilities.mean(dim=0)
    p_pred_sensitive = joint_class_probabilities.mean(dim=0)
    entropy_sensitive = entropy(p_sensitive)
    entropy_target = entropy(p_pred)
    joint_entropy = entropy(p_pred_sensitive)
    mi = entropy_sensitive + entropy_target - joint_entropy
    return mi


def fair_mutual_information_eo_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    target_attribute_targets = multi_attribute_targets[:, target_attribute.index]
    pred_target_class_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    sensitive_class_probabilities = one_hot(
        multi_attribute_targets[:, sensitive_attribute.index], num_classes=sensitive_attribute.size
    ).float()
    joint_class_probabilities = sensitive_class_probabilities.unsqueeze(
        dim=2
    ) * pred_target_class_probabilities.unsqueeze(dim=1)
    mi = 0.0
    for target_class_a in range(target_attribute.size):
        from_target_class_a = target_attribute_targets.eq(target_class_a)
        if from_target_class_a.sum() == 0:
            print(f"no samples from target class a ({target_class_a})")
            return tensor(0.0, device=multi_attribute_targets.device)

        p_sensitive = sensitive_class_probabilities[from_target_class_a].mean(dim=0)
        p_pred = pred_target_class_probabilities[from_target_class_a].mean(dim=0)
        p_pred_sensitive_target = joint_class_probabilities[from_target_class_a].mean(dim=0)
        entropy_sensitive = entropy(p_sensitive)
        entropy_target = entropy(p_pred)
        joint_entropy = entropy(p_pred_sensitive_target)
        mi = mi + entropy_sensitive + entropy_target - joint_entropy
    return mi
