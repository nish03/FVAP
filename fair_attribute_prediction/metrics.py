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
        state.additional_loss_totals[additional_loss_name] += additional_loss * prediction_count
    state.correct_prediction_counts += multi_attribute_predictions.eq(multi_attribute_targets).sum(dim=0)

    attribute_accuracies = 100 * state.correct_prediction_counts / state.processed_prediction_count
    metric_accuracy = attribute_accuracies.mean().item()
    metric_loss = state.optimized_loss_total / state.processed_prediction_count
    _metrics = {
        "loss": metric_loss,
        "accuracy": metric_accuracy,
    }
    for additional_loss_name, additional_loss in state.additional_loss_totals.items():
        metric_loss = state.additional_loss_totals[additional_loss_name] / state.processed_prediction_count
        _metrics[f"additional_loss_{additional_loss_name}"] = metric_loss

    return _metrics, state
