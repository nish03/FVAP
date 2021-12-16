from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import no_grad, zeros

from Util import get_device


@dataclass
class MetricsState:
    processed_prediction_count = 0.0
    loss_total = 0.0
    loss_term_totals = None
    correct_prediction_counts = None


@no_grad()
def metrics(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    loss_value: float,
    loss_term_values: Dict[str, float],
    state: Optional[MetricsState] = None,
):
    multi_attribute_predictions = model.module.multi_attribute_predictions(multi_output_class_logits)
    prediction_count = multi_attribute_targets.shape[0]
    if state is None:
        state = MetricsState()
        state.loss_term_totals = defaultdict(float)
        state.correct_prediction_counts = zeros(multi_attribute_targets.shape[1], device=get_device())

    state.processed_prediction_count += prediction_count
    state.loss_total += loss_value * prediction_count
    for loss_term_name in loss_term_values:
        state.loss_term_totals[loss_term_name] += loss_term_values[loss_term_name] * prediction_count
    state.correct_prediction_counts += multi_attribute_predictions.eq(multi_attribute_targets).sum(dim=0)

    attribute_accuracies = 100 * state.correct_prediction_counts / state.processed_prediction_count
    accuracy = attribute_accuracies.mean().item()
    loss = state.loss_total / state.processed_prediction_count
    _metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    for loss_term_name in state.loss_term_totals:
        loss_term = state.loss_term_totals[loss_term_name] / state.processed_prediction_count
        _metrics[f"loss_term_{loss_term_name}"] = loss_term
    for attribute_index in range(len(attribute_accuracies)):
        attribute_accuracy = attribute_accuracies[attribute_index].item()
        _metrics[f"attribute_{attribute_index}_accuracy"] = attribute_accuracy
    return _metrics, state
