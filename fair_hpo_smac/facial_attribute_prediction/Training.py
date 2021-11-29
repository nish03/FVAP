from collections import defaultdict

from torch import zeros, no_grad
from tqdm.notebook import tqdm

from Util import get_device
from Evaluation import EvaluationState, evaluate


def train_classifier(
    parallel_model,
    optimizer,
    epoch_count,
    train_dataloader,
    valid_dataloader,
):
    model = parallel_model.module
    epoch_history = defaultdict(list)
    best_state = {
        "valid_accuracy": 0.0,
    }
    device = get_device()
    train_dataset = train_dataloader.dataset
    valid_dataset = valid_dataloader.dataset
    for epoch in range(1, epoch_count + 1):
        epoch_description = f"Epoch {epoch:0>2}/{epoch_count:0>2}"

        model.train()
        train_eval_state = EvaluationState()
        train_eval_state.correct_prediction_counts = zeros(
            train_dataset.attribute_count, device=device
        )
        with tqdm(train_dataloader, unit="batch") as train_epoch_iterator:
            train_epoch_iterator.set_description(f"{epoch_description} Train")
            for batch_index, data in enumerate(train_epoch_iterator):
                images, class_index_targets = data[0].to(device), data[1].to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = parallel_model(images)
                class_index_predictions = model.predict(outputs)

                loss = model.criterion(outputs, class_index_targets)
                loss.backward()

                optimizer.step()

                train_eval_results = evaluate(
                    class_index_predictions,
                    class_index_targets,
                    loss,
                    train_eval_state,
                )
                train_epoch_iterator.set_postfix(
                    loss=train_eval_results["loss"],
                    accuracy=train_eval_results["accuracy"],
                )
        epoch_history["train"].append(train_eval_results)

        parallel_model.eval()
        valid_eval_state = EvaluationState()
        valid_eval_state.correct_prediction_counts = zeros(
            valid_dataset.attribute_count, device=device
        )
        with tqdm(valid_dataloader, unit="batch") as valid_epoch_iterator, no_grad():
            valid_epoch_iterator.set_description(f"{epoch_description} Valid")
            for batch_index, data in enumerate(valid_epoch_iterator):
                images, class_index_targets = data[0].to(device), data[1].to(device)

                outputs = parallel_model(images)
                class_index_predictions = model.predict(outputs)

                loss = model.criterion(outputs, class_index_targets)

                valid_eval_results = evaluate(
                    class_index_predictions,
                    class_index_targets,
                    loss,
                    valid_eval_state,
                )

                valid_epoch_iterator.set_postfix(
                    loss=valid_eval_results["loss"],
                    accuracy=valid_eval_results["accuracy"],
                )
        epoch_history["valid"].append(valid_eval_results)
        if best_state["valid_accuracy"] < valid_eval_results["accuracy"]:
            best_state["valid_accuracy"] = valid_eval_results["accuracy"]
            best_state["model_state_dict"] = model.state_dict()
            best_state["optimizer_state_dict"] = optimizer.state_dict()
            best_state["epoch"] = epoch

    return epoch_history, best_state
