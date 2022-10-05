from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import no_grad, zeros

from util import get_device


@dataclass
class MetricsState:
    """
    State of the :func:`metrics()` computation during training and validation (intertwined)
    """
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
    """
    Computes metrics by accumulating statistics during training and validation (intertwined).

    :param model: Module / MultiAttributeClassifier predicting the logits for each attribute
    :param multi_output_class_logits: List(Tensor[sample_count, attribute_count(, attribute_class_count)])
        containing as many tensors as there are unique attribute sizes.
        All Tensors contain the predicted logits for each sample, attribute of the given size and attribute class.
        The tensor for binary attributes has only 2 dimensions as the logits for the other class can be deduced.
    :param multi_attribute_targets: Tensor([sample_count, overall_attribute_size])
        containing the ground truth labels for each sample and prediction attribute
    :param optimized_loss: Optimized loss value,
    :param additional_losses: Dict[str, float] maps additional fair loss names to their values
    :param state: (Optional) MetricsState storing the accumulated state of the metric computation.
        This function will return an initialized state if this parameter is set to None.
    :return: Dict[str, float] maps each metric name to its value,
             MetricState contains the updated/initialized version of state
    """
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
