from typing import List

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy


def cross_entropy_loss(
    model: torch.nn.Module, multi_output_class_logits: List[torch.Tensor], multi_attribute_targets: torch.Tensor
) -> torch.Tensor:
    output_losses = []
    for attribute_size, output_class_logits in zip(model.module.unique_attribute_sizes, multi_output_class_logits):
        output_attribute_targets = multi_attribute_targets[:, model.module.attribute_sizes.eq(attribute_size)]
        if attribute_size == 2:
            output_attribute_targets = output_attribute_targets.float()
            output_loss = binary_cross_entropy_with_logits(
                output_class_logits, output_attribute_targets, reduction="sum"
            )
        else:
            output_loss = cross_entropy(output_class_logits, output_attribute_targets, reduction="sum")
        output_losses.append(output_loss)

    _cross_entropy_loss = sum(output_losses) / multi_attribute_targets.numel()
    return _cross_entropy_loss
