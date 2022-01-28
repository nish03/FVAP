from typing import Dict, Tuple

import comet_ml
import torch.utils.data

from losses.loss import loss_with_metrics
from util import get_learning_rate


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    parameters: Dict,
    experiment: comet_ml.Experiment,
) -> Tuple[Dict, Dict]:
    best_model_state = {}
    best_valid_loss = None
    epoch_count = parameters["epoch_count"]
    sensitive_attribute = train_dataloader.dataset.attribute(parameters["sensitive_attribute_index"])
    target_attribute = train_dataloader.dataset.attribute(parameters["target_attribute_index"])
    fair_loss_type = parameters["fair_loss_type"]
    fair_loss_weight = parameters["fair_loss_weight"]
    metrics_averaging_weight = parameters["metrics_averaging_weight"]
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_metrics_state = None
        with experiment.context_manager("train"):
            for batch_data in train_dataloader:
                optimizer.zero_grad(set_to_none=True)

                _loss, train_metrics, train_metrics_state = loss_with_metrics(
                    model,
                    batch_data,
                    train_metrics_state,
                    metrics_averaging_weight,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )

                _loss.backward()
                optimizer.step()

            experiment.log_metrics(train_metrics, epoch=epoch)

        model.eval()
        valid_metrics_state = None
        with experiment.context_manager("valid"):
            for batch_data in valid_dataloader:
                _loss, valid_metrics, valid_metrics_state = loss_with_metrics(
                    model,
                    batch_data,
                    valid_metrics_state,
                    metrics_averaging_weight,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )
            experiment.log_metrics(valid_metrics, epoch=epoch)
        averaged_valid_loss = valid_metrics["averaged_loss"]
        if best_valid_loss is None or best_valid_loss < averaged_valid_loss:
            best_valid_loss = averaged_valid_loss
            best_model_state = {
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

        if parameters["learning_rate_scheduler"] == "ReduceLROnPlateau":
            lr_scheduler.step(averaged_valid_loss)

        experiment.log_metric("learning_rate", get_learning_rate(optimizer), epoch=epoch)

    final_model_state = {
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
