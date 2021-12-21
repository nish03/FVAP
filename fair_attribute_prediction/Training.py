from typing import Dict, Tuple

import comet_ml
import torch.utils.data

from Loss import loss_with_metrics


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_evaluation_state = None
        with experiment.context_manager("train"):
            for batch_data in train_dataloader:
                optimizer.zero_grad(set_to_none=True)

                _loss, train_metrics, train_evaluation_state = loss_with_metrics(
                    model,
                    batch_data,
                    train_evaluation_state,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )

                _loss.backward()
                optimizer.step()

            experiment.log_metrics(train_metrics, epoch=epoch)

        model.eval()
        valid_evaluation_state = None
        with experiment.context_manager("valid"):
            for batch_data in valid_dataloader:
                _loss, valid_metrics, valid_evaluation_state = loss_with_metrics(
                    model,
                    batch_data,
                    valid_evaluation_state,
                    sensitive_attribute,
                    target_attribute,
                    fair_loss_type,
                    fair_loss_weight,
                )
            experiment.log_metrics(valid_metrics, epoch=epoch)
        epoch_valid_loss = valid_metrics["loss"]
        if best_valid_loss is None or best_valid_loss < epoch_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_state = {
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

    final_model_state = {
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
