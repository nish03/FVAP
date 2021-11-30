from torch import no_grad
from dataclasses import dataclass


@dataclass
class EvaluationState:
    processed_prediction_count = 0.0
    loss_total = 0.0
    correct_prediction_counts = None


@no_grad()
def evaluate(
    class_index_predictions,
    class_index_targets,
    loss_value,
    evaluation_state,
):
    print(f"{class_index_predictions.shape=}")
    print(f"{class_index_targets.shape=}")
    print(f"{evaluation_state.correct_prediction_counts.shape=}")
    prediction_count = len(class_index_targets)

    evaluation_state.processed_prediction_count += prediction_count
    evaluation_state.loss_total += loss_value.item() * prediction_count
    evaluation_state.correct_prediction_counts += (
        class_index_predictions == class_index_targets
    ).sum(dim=0)

    loss = evaluation_state.loss_total / evaluation_state.processed_prediction_count
    attribute_accuracies = (
        100
        * evaluation_state.correct_prediction_counts
        / evaluation_state.processed_prediction_count
    )
    accuracy = attribute_accuracies.mean().item()

    return {
        "loss": loss,
        "attribute_accuracies": attribute_accuracies,
        "accuracy": accuracy,
    }
