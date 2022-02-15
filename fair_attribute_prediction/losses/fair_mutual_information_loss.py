import torch
from torch import nonzero
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    nonzero_probabilities = probabilities[nonzero(probabilities, as_tuple=True)]
    return -(nonzero_probabilities * nonzero_probabilities.log()).sum()


def fair_mutual_information_loss(
    model: torch.nn.Module,
    multi_output_class_logits: torch.Tensor,
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    batch_target_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )
    batch_sensitive_probabilities = one_hot(multi_attribute_targets[:, sensitive_attribute.index]).float()
    batch_joint_probabilities = batch_sensitive_probabilities.unsqueeze(dim=2) * batch_target_probabilities.unsqueeze(
        dim=1
    )
    sensitive_probabilities = batch_sensitive_probabilities.mean(dim=0)
    target_probabilities = batch_target_probabilities.mean(dim=0)
    joint_probabilities = batch_joint_probabilities.mean(dim=0)
    sensitive_entropy = entropy(sensitive_probabilities)
    target_entropy = entropy(target_probabilities)
    joint_entropy = entropy(joint_probabilities)
    mi = sensitive_entropy + target_entropy - joint_entropy
    return mi
