from collections import defaultdict
from dataclasses import dataclass

from torch import no_grad, zeros

from Util import get_device


@dataclass
class EvaluationState:
    processed_prediction_count = 0.0
    loss_total = 0.0
    loss_term_totals = None
    correct_prediction_counts = None


@no_grad()
def evaluate(
    class_index_predictions,
    class_index_targets,
    loss_value,
    loss_term_values,
    state=None,
):
    prediction_count = class_index_targets.shape[0]
    if state is None:
        state = EvaluationState()
        state.loss_term_totals = defaultdict(float)
        state.correct_prediction_counts = zeros(
            class_index_targets.shape[1], device=get_device()
        )

    state.processed_prediction_count += prediction_count
    state.loss_total += loss_value.item() * prediction_count
    for loss_term_name in loss_term_values:
        state.loss_term_totals[loss_term_name] += (
            loss_term_values[loss_term_name].item() * prediction_count
        )
    state.correct_prediction_counts += (
        class_index_predictions == class_index_targets
    ).sum(dim=0)

    results = {
        "loss": state.loss_total / state.processed_prediction_count,
        "attribute_accuracies": (
            100 * state.correct_prediction_counts / state.processed_prediction_count
        ).cpu().numpy(),
    }
    for loss_term_name in state.loss_term_totals:
        results[f"loss_term_{loss_term_name}"] = (
            state.loss_term_totals[loss_term_name] / state.processed_prediction_count
        )
    results["accuracy"] = results["attribute_accuracies"].mean().item()

    return results, state
