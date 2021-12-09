import comet_ml
import torch.utils.data

from Evaluation import evaluate
from Util import get_device


def train_classifier(
    model_parallel: torch.nn.DataParallel,
    optimizer: torch.optim.Optimizer,
    epoch_count: int,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    experiment: comet_ml.Experiment,
):
    best_model_state = {}
    model = model_parallel.module
    best_valid_loss = None
    device = get_device()
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_eval_state = None
        with experiment.context_manager("train"):
            for data in train_dataloader:
                images, class_index_targets = data[0].to(device), data[1].to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = model_parallel(images)
                class_index_predictions = model.predict(outputs)

                loss, loss_terms = model.criterion(outputs, class_index_targets)
                loss.backward()

                optimizer.step()

                train_eval_results, train_eval_state = evaluate(
                    class_index_predictions,
                    class_index_targets,
                    loss,
                    loss_terms,
                    train_eval_state,
                )
            experiment.log_metrics(train_eval_results, epoch=epoch)

        model_parallel.eval()
        valid_eval_state = None
        with experiment.context_manager("valid"):
            for data in valid_dataloader:
                images, class_index_targets = data[0].to(device), data[1].to(device)

                outputs = model_parallel(images)
                class_index_predictions = model.predict(outputs)

                loss, loss_terms = model.criterion(outputs, class_index_targets)

                valid_eval_results, valid_eval_state = evaluate(
                    class_index_predictions,
                    class_index_targets,
                    loss,
                    loss_terms,
                    valid_eval_state,
                )

            experiment.log_metrics(valid_eval_results, epoch=epoch)
        epoch_valid_loss = valid_eval_results["loss"]
        if best_valid_loss is None or best_valid_loss < epoch_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_state = {
                "valid_results": valid_eval_results,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

    final_model_state = {
        "valid_results": valid_eval_results,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
