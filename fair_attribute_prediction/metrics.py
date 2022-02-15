from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import no_grad, zeros

from util import get_device


@dataclass
class MetricsState:
    processed_prediction_count = 0.0
    optimized_loss_total = 0.0
    additional_loss_totals = None
    correct_prediction_counts = None


@no_grad()
def metrics(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    optimized_loss: float,
    additional_losses: Dict[str, float],
    state: Optional[MetricsState] = None,
) -> (Dict[str, float], MetricsState):
    multi_attribute_predictions = model.module.multi_attribute_predictions(multi_output_class_logits)
    prediction_count = multi_attribute_targets.shape[0]
    if state is None:
        state = MetricsState()
        state.additional_loss_totals = defaultdict(float)
        state.correct_prediction_counts = zeros(multi_attribute_targets.shape[1], device=get_device())

    state.processed_prediction_count += prediction_count
    state.optimized_loss_total += optimized_loss * prediction_count
    for additional_loss_name, additional_loss in additional_losses.items():
        state.additional_loss_totals[additional_loss_name, additional_loss] += additional_loss * prediction_count
    state.correct_prediction_counts += multi_attribute_predictions.eq(multi_attribute_targets).sum(dim=0)

    attribute_accuracies = 100 * state.correct_prediction_counts / state.processed_prediction_count
    accuracy = attribute_accuracies.mean().item()
    loss = state.optimized_loss_total / state.processed_prediction_count
    _metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    for additional_loss_name, additional_loss in state.additional_loss_totals:
        loss_term = (
            state.additional_loss_totals[additional_loss_name, additional_loss] / state.processed_prediction_count
        )
        _metrics[f"additional_loss_{additional_loss_name}"] = loss_term

    return _metrics, state


def averaged_metrics(
    current_metrics, previous_averaged_metrics: Optional[Dict[str, float]] = None, averaging_weight: float = 0.5
):
    _averaged_metrics = {}
    for metric_name in current_metrics:
        averaged_metric_name = f"averaged_{metric_name}"
        _averaged_metrics[averaged_metric_name] = (
            previous_averaged_metrics[averaged_metric_name] * averaging_weight
            + current_metrics[metric_name] * (1 - averaging_weight)
            if previous_averaged_metrics is not None
            else current_metrics[metric_name]
        )

    return _averaged_metrics
