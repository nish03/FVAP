import torch
from torch import nonzero
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    nonzero_probabilities = probabilities[nonzero(probabilities, as_tuple=True)]
    return -(nonzero_probabilities * nonzero_probabilities.log()).sum()


def fair_mutual_information_3_loss(
    model: torch.nn.Module,
    multi_output_class_logits: torch.Tensor,
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    sensitive_real_probabilities = one_hot(
        multi_attribute_targets[:, sensitive_attribute.index], num_classes=sensitive_attribute.size
    ).float()  # p(a* | x, θ)
    target_real_probabilities = one_hot(
        multi_attribute_targets[:, target_attribute.index], num_classes=target_attribute.size
    ).float()  # p(y* | x, θ)
    target_predicted_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_attribute.index
    )  # p(y | x, θ)
    joint_probabilities = (
        sensitive_real_probabilities.reshape([-1, sensitive_attribute.size, 1, 1])
        * target_real_probabilities.reshape([-1, 1, target_attribute.size, 1])
        * target_predicted_probabilities.reshape([-1, 1, 1, target_attribute.size])
    ).mean(
        dim=0
    )  # p(a*, y*, y | θ)
    sensitive_probabilities = joint_probabilities.sum(dim=[1, 2])  # p(a* | θ)
    target_probabilities = joint_probabilities.sum(dim=0)  # p(a* | θ)
    sensitive_entropy = entropy(sensitive_probabilities)
    target_entropy = entropy(target_probabilities)
    joint_entropy = entropy(joint_probabilities)
    print(f"{sensitive_entropy=} {target_entropy=} {joint_entropy=}")
    mi = sensitive_entropy + target_entropy - joint_entropy
    return mi


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
