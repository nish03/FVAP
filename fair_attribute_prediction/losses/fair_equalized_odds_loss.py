from typing import List

import torch

from multi_attribute_dataset import Attribute


def fair_equalized_odds_loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
) -> torch.Tensor:
    pass
