from typing import Dict, List, Optional, Tuple
import torch
from torch import no_grad

from losses.cross_entropy_loss import cross_entropy_loss
from losses.fair_losses import fair_losses
from metrics import MetricsState, metrics
from multi_attribute_dataset import Attribute
from util import get_device


def losses(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    optimized_fair_loss_type: str,
    optimized_fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    optimized_cross_entropy_loss = cross_entropy_loss(model, multi_output_class_logits, multi_attribute_targets)

    optimized_fair_loss = fair_losses[optimized_fair_loss_type](
        model, multi_output_class_logits, multi_attribute_targets, sensitive_attribute, target_attribute
    )

    optimized_loss = optimized_cross_entropy_loss + optimized_fair_loss_weight * optimized_fair_loss

    with no_grad():
        additional_losses = {
            "cross_entropy": optimized_cross_entropy_loss.item(),
            f"fair_{optimized_fair_loss_type}": optimized_fair_loss.item(),
        }

        for additional_fair_loss_type in fair_losses:
            if additional_fair_loss_type == optimized_fair_loss_type:
                continue
            additional_fair_loss = fair_losses[additional_fair_loss_type](
                model, multi_output_class_logits, multi_attribute_targets, sensitive_attribute, target_attribute
            )
            additional_losses[f"fair_{additional_fair_loss_type}"] = additional_fair_loss.item()

    return optimized_loss, additional_losses


def losses_with_metrics(
    model: torch.nn.Module,
    batch_data: (torch.Tensor, torch.Tensor),
    metrics_state: Optional[MetricsState],
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    optimized_fair_loss_type: str,
    optimized_fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], MetricsState]:
    images, multi_attribute_targets = batch_data[0].to(get_device()), batch_data[1].to(get_device())

    multi_output_class_logits = model(images)

    optimized_loss, additional_losses = losses(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        sensitive_attribute,
        target_attribute,
        optimized_fair_loss_type,
        optimized_fair_loss_weight,
    )

    _metrics, metrics_state = metrics(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        optimized_loss.item(),
        additional_losses,
        metrics_state,
    )
    return optimized_loss, _metrics, metrics_state
