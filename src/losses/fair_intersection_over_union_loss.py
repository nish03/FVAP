from itertools import combinations

import torch
from torch import tensor, ones_like
from torch.nn.functional import one_hot

from multi_attribute_dataset import Attribute


def intersection_over_union(
    class_probabilities: torch.Tensor,
    attribute_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Approximates the intersection over union from attribute class probabilities and ground truth labels.

    :param class_probabilities: Tensor[sample_count, class_count]
        containing the predicted probabilities for each sample and attribute class
    :param attribute_targets: Tensor[sample_count]
        containing the ground truth labels for each sample and attribute class
    :return: Tensor[] containing the differentiable intersection over union value
    """
    attribute_size = class_probabilities.shape[1]
    confusion_matrix = (
        class_probabilities.unsqueeze(dim=2) * one_hot(attribute_targets, attribute_size).unsqueeze(dim=1)
    ).sum(dim=0)
    iou = 0.0
    for attribute_class in range(attribute_size):
        iou = iou + confusion_matrix[attribute_class, attribute_class] / (
            confusion_matrix[attribute_class, :].sum()
            + confusion_matrix[:, attribute_class].sum()
            - confusion_matrix[attribute_class, attribute_class]
        )
    iou /= attribute_size
    return iou


def sensitive_intersection_over_union(
    from_sensitive_class: torch.Tensor, target_class_probabilities: torch.Tensor, target_attribute_targets: torch.Tensor
) -> torch.Tensor:
    """
    Approximates the intersection over union from target attribute class probabilities and ground truth labels
    conditioned on a sensitive attribute class.

    :param from_sensitive_class: Tensor[sample_count]
        containing booleans that indicate if the sample belongs to the conditioned sensitive class
    :param target_class_probabilities: Tensor[sample_count, class_count]
        containing the predicted probabilities for each sample and target attribute class
    :param target_attribute_targets: Tensor[sample_count]
        containing the ground truth labels for each sample and attribute class
    :return: Tensor[] containing the differentiable intersection over union value
    """
    class_probabilities = target_class_probabilities[from_sensitive_class]
    attribute_targets = target_attribute_targets[from_sensitive_class]
    sensitive_iou = intersection_over_union(class_probabilities, attribute_targets)
    return sensitive_iou


def fair_intersection_over_union_paired_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    """
    Computes the fair paired intersection over union loss from sensitive attribute labels, target attribute labels
    and target attribute class probabilities.

    The fairness loss value is based on the sum of squared differences between the IoUs conditioned on all combinations
    of different attribute classes.

    :param sensitive_attribute: Sensitive Attribute
        with targets member (Tensor[sample_count]) containing class labels of each sample
    :param target_attribute: Target Attribute
        with targets member (Tensor[sample_count]) containing class labels of each sample and
        with class_probabilities member (Tensor[sample_count, class_count]) containing predicted probabilities of each
        sample and class
    :return: Tensor[] containing the differentiable loss value
    """
    fair_iou_loss = 0.0
    for sensitive_class_a, sensitive_class_b in combinations(range(sensitive_attribute.size), 2):
        from_sensitive_class_a = sensitive_attribute.targets.eq(sensitive_class_a)
        from_sensitive_class_b = sensitive_attribute.targets.eq(sensitive_class_b)
        if from_sensitive_class_a.sum() == 0 or from_sensitive_class_b.sum() == 0:
            return tensor(0.0, device=sensitive_attribute.targets.device)
        iou_a = sensitive_intersection_over_union(
            from_sensitive_class_a, target_attribute.class_probabilities, target_attribute.targets
        )
        iou_b = sensitive_intersection_over_union(
            from_sensitive_class_b, target_attribute.class_probabilities, target_attribute.targets
        )
        fair_iou_loss = fair_iou_loss + (iou_a - iou_b).pow(2)

    fair_iou_loss /= sensitive_attribute.size * (sensitive_attribute.size - 1) / 2
    return fair_iou_loss


def fair_intersection_over_union_conditioned_loss(
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    """
    Computes the fair conditioned intersection over union loss from sensitive attribute labels, target attribute labels
    and target attribute class probabilities.

    The fairness loss value is based on the sum of squared differences between the IoUs conditioned on each attribute
    class and the general IoU.

    :param sensitive_attribute: Sensitive Attribute
        with targets member (Tensor[sample_count]) containing class labels of each sample
    :param target_attribute: Target Attribute
        with targets member (Tensor[sample_count]) containing class labels of each sample and
        with class_probabilities member (Tensor[sample_count, class_count]) containing predicted probabilities of each
        sample and class
    :return: Tensor[] containing the differentiable loss value
    """
    fair_iou_loss = 0.0
    for sensitive_class_a in range(sensitive_attribute.size):
        from_sensitive_class_a = sensitive_attribute.targets.eq(sensitive_class_a)
        from_any_sensitive_class = ones_like(from_sensitive_class_a)
        if from_sensitive_class_a.sum() == 0:
            return tensor(0.0, device=sensitive_attribute.targets.device)
        iou_a = sensitive_intersection_over_union(
            from_sensitive_class_a, target_attribute.class_probabilities, target_attribute.targets
        )
        iou_general = sensitive_intersection_over_union(
            from_any_sensitive_class, target_attribute.class_probabilities, target_attribute.targets
        )
        fair_iou_loss = fair_iou_loss + (iou_a - iou_general).pow(2)

    fair_iou_loss /= sensitive_attribute.size
    return fair_iou_loss
