from typing import List

import torch
from torch import stack
from torch.nn.functional import cross_entropy


def cross_entropy_loss(
    model: torch.nn.Module, multi_output_class_logits: List[torch.Tensor], multi_attribute_targets: torch.Tensor
) -> torch.Tensor:
    """
    Computes the cross entropy loss between the predicted multi attribute logits and the target labels
    :param model: :class:`torch.nn.Module` / :class:`MultiAttributeClassifier`
        predicting the logits for each attribute
    :param multi_output_class_logits: List(Tensor[sample_count, attribute_count(, attribute_class_count)])
        containing as many tensors as there are unique prediction attribute sizes.
        All Tensors contain the predicted logits for each sample, attribute of the given size and attribute class.
        The tensor for binary attributes has only 2 dimensions as the logits for the other class can be deduced.
    :param multi_attribute_targets: Tensor([sample_count, overall_attribute_size])
        containing the ground truth labels for each sample and prediction attribute
    :return: Tensor[] containing the differentiable loss value
    """
    output_losses = []
    for attribute_size, output_class_logits in zip(model.module.unique_attribute_sizes, multi_output_class_logits):
        attribute_size_matches = model.module.attribute_sizes.eq(attribute_size)
        attribute_indices = attribute_size_matches.eq(1).nonzero()
        for output_class_logits_index, attribute_index in enumerate(attribute_indices):
            attribute_class_logits = output_class_logits[..., output_class_logits_index]
            attribute_targets = multi_attribute_targets[:, attribute_index].flatten()
            attribute_class_weights = model.module.attribute_class_weights[attribute_index]
            if attribute_class_weights is not None:
                attribute_class_weights = attribute_class_weights.to(attribute_class_logits.device)
            if attribute_size == 2:
                attribute_class_logits = stack([-attribute_class_logits, attribute_class_logits], dim=-1)
            output_loss = cross_entropy(attribute_class_logits, attribute_targets, weight=attribute_class_weights)
            output_losses.append(output_loss)

    _cross_entropy_loss = sum(output_losses) / multi_attribute_targets.shape[1]
    return _cross_entropy_loss
