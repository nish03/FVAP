from typing import Dict, Tuple

import comet_ml
import torch.utils.data

from losses.loss import losses_with_metrics
from metrics import averaged_metrics
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
    best_averaged_valid_loss = None
    epoch_count = parameters["epoch_count"]
    sensitive_attribute = train_dataloader.dataset.attribute(parameters["sensitive_attribute_index"])
    target_attribute = train_dataloader.dataset.attribute(parameters["target_attribute_index"])
    fair_loss_type = parameters["fair_loss_type"]
    fair_loss_weight = parameters["fair_loss_weight"]
    metrics_averaging_weight = parameters["metrics_averaging_weight"]
    epoch_train_metrics = None
    epoch_valid_metrics = None
    for epoch in range(1, epoch_count + 1):

        model.train()
        with experiment.train():
            train_metrics_state = None

            for batch_data in train_dataloader:
                optimizer.zero_grad(set_to_none=True)

                optimized_loss, batch_train_metrics, train_metrics_state = losses_with_metrics(
                    model,
                    batch_data,
                    train_metrics_state,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )

                optimized_loss.backward()
                optimizer.step()

            averaged_train_metrics = averaged_metrics(
                batch_train_metrics, epoch_train_metrics, metrics_averaging_weight
            )
            epoch_train_metrics = batch_train_metrics
            epoch_train_metrics.update(averaged_train_metrics)

            experiment.log_metrics(epoch_train_metrics, epoch=epoch)

        model.eval()
        with experiment.validate():
            valid_metrics_state = None

            for batch_data in valid_dataloader:
                optimized_loss, batch_valid_metrics, valid_metrics_state = losses_with_metrics(
                    model,
                    batch_data,
                    valid_metrics_state,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )
            averaged_valid_metrics = averaged_metrics(
                batch_valid_metrics, epoch_valid_metrics, metrics_averaging_weight
            )
            epoch_valid_metrics = batch_valid_metrics
            epoch_valid_metrics.update(averaged_valid_metrics)

            experiment.log_metrics(epoch_valid_metrics, epoch=epoch)

        averaged_valid_loss = epoch_valid_metrics["averaged_loss"]
        if best_averaged_valid_loss is None or best_averaged_valid_loss < averaged_valid_loss:
            best_averaged_valid_loss = averaged_valid_loss
            best_model_state = {
                "train_metrics": epoch_train_metrics,
                "valid_metrics": epoch_valid_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

        if parameters["learning_rate_scheduler"] == "reduce_lr_on_plateau":
            lr_scheduler.step(averaged_valid_loss)

        experiment.log_metric("learning_rate", get_learning_rate(optimizer), epoch=epoch)

    final_model_state = {
        "train_metrics": epoch_train_metrics,
        "valid_metrics": epoch_valid_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
