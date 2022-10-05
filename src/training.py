from typing import Dict, Tuple, OrderedDict

import comet_ml
import torch.utils.data
from torch import no_grad, Tensor
from torch.cuda import empty_cache

from losses.loss import losses_with_metrics
from util import get_learning_rate


def model_state_dict(model: torch.nn.Module, device: str = "cpu") -> OrderedDict[str, Tensor]:
    """
    Gets the state dict of a module stored on a given device.
    :param model: Module
    :param device: Device for storage
    :return: OrderedDict[str, Tensor] maps names to weight Tensors
    """
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = value.to(device)
    return state_dict


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    parameters: Dict,
    experiment: comet_ml.Experiment,
) -> Tuple[Dict, Dict]:
    """
    Trains a fair classifier.

    :param model: Module / MultiAttributeClassifier predicting the logits for each attribute
    :param optimizer: Optimizer
    :param lr_scheduler: Learning rate scheduler
    :param train_dataloader: DataLoader for the train dataset split
    :param valid_dataloader: DataLoader for the validation dataset split
    :param parameters: Dict mapping hyperparameter names to values (see :func:`run_experiment`)
    :param experiment: comet_ml.Experiment
    :return: Dict State of the model with the lowest validation loss during the entire training procedure,
             Dict State of the final model after the entire training procedure was performed
        States consists of training metrics ("train_metrics"), validation metrics ("valid_metrics"),
        model state("model_state_dict"), optimizer state ("optimizer_state_dict") and the corresponding epoch ("epoch").
    """
    best_model_state = {}
    best_valid_loss = None

    epoch_count = parameters["epoch_count"]
    sensitive_attribute = train_dataloader.dataset.attribute(parameters["sensitive_attribute_index"])
    target_attribute = train_dataloader.dataset.attribute(parameters["target_attribute_index"])
    prediction_attribute_indices = train_dataloader.dataset.prediction_attribute_indices
    fair_loss_type = parameters["fair_loss_type"]
    fair_loss_weight = parameters["fair_loss_weight"]
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
                    prediction_attribute_indices,
                    fair_loss_type,
                    fair_loss_weight,
                )

                optimized_loss.backward()
                optimizer.step()

            epoch_train_metrics = batch_train_metrics

            experiment.log_metrics(epoch_train_metrics, epoch=epoch)

            del batch_data
            del train_metrics_state
            del optimized_loss
            empty_cache()

        model.eval()
        with experiment.validate():
            valid_metrics_state = None

            with no_grad():
                for batch_data in valid_dataloader:
                    optimized_loss, batch_valid_metrics, valid_metrics_state = losses_with_metrics(
                        model,
                        batch_data,
                        valid_metrics_state,
                        sensitive_attribute,
                        target_attribute,
                        prediction_attribute_indices,
                        fair_loss_type,
                        fair_loss_weight,
                    )

            epoch_valid_metrics = batch_valid_metrics

            experiment.log_metrics(epoch_valid_metrics, epoch=epoch)

            del optimized_loss
            del valid_metrics_state
            del batch_data
            empty_cache()

        valid_loss = epoch_valid_metrics["loss"]
        if best_valid_loss is None or best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            best_model_state = {
                "train_metrics": epoch_train_metrics,
                "valid_metrics": epoch_valid_metrics,
                "model_state_dict": model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

        if parameters["learning_rate_scheduler"] == "reduce_lr_on_plateau":
            valid_loss = epoch_valid_metrics["loss"]
            lr_scheduler.step(valid_loss)

        experiment.log_metric("learning_rate", get_learning_rate(optimizer), epoch=epoch)

    final_model_state = {
        "train_metrics": epoch_train_metrics,
        "valid_metrics": epoch_valid_metrics,
        "model_state_dict": model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
