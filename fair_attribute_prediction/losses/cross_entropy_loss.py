from typing import List

import torch
from torch import tensor, stack
from torch.nn.functional import cross_entropy


def cross_entropy_loss(
    model: torch.nn.Module, multi_output_class_logits: List[torch.Tensor], multi_attribute_targets: torch.Tensor
) -> torch.Tensor:
    output_losses = []
    for attribute_size, output_class_logits in zip(model.module.unique_attribute_sizes, multi_output_class_logits):
        attribute_size_matches = model.module.attribute_sizes.eq(attribute_size)
        attribute_indices = attribute_size_matches.eq(1).nonzero()
        for output_class_logits_index, attribute_index in enumerate(attribute_indices):
            attribute_class_logits = output_class_logits[..., output_class_logits_index]
            attribute_targets = multi_attribute_targets[:, attribute_index].flatten()
            attribute_class_weights = tensor(
                model.module.attribute_class_weights[attribute_index],
                device=output_class_logits.device,
            )
            if attribute_size == 2:
                attribute_class_logits = stack([-attribute_class_logits, attribute_class_logits], dim=-1)
            output_loss = cross_entropy(
                attribute_class_logits, attribute_targets, weight=attribute_class_weights, reduction="sum"
            )
            output_losses.append(output_loss)

    _cross_entropy_loss = sum(output_losses) / multi_attribute_targets.numel()
    return _cross_entropy_loss
