import traceback
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict

from comet_ml import Experiment
from comet_ml.experiment import BaseExperiment
from torch import save
from torch.backends import cudnn

from losses.fair_losses import fair_losses
from training import train_classifier
from evaluation import evaluate_classifier
from util import create_dataset, create_dataloader, create_model, create_optimizer, create_lr_scheduler


def log_experiment_status(experiment: BaseExperiment, status: str):
    """
    Logs an experiment status to the command line and to the CometML notification service.

    :param experiment: comet_ml.Experiment
    :param status: Status message
    """
    status_message = f"Experiment '{experiment.get_name()}' {status}"
    print(status_message)
    experiment.send_notification(status_message)


def fair_attribute_prediction_experiment(parameters: Dict, experiment_name: str) -> (BaseExperiment, Dict, Dict):
    """
    Performs a fair attribute prediction experiment.

    :param parameters: Dict containing experiment hyperparameters
    :param experiment_name: Name of the experiment
    :return: comet_ml.Experiment that was performed,
             Dict containing the best model state that was returned from train_classifier(),
             Dict containing the final model state that was returned from train_classifier()
    """
    start_date = datetime.utcnow()
    experiment = Experiment(
        project_name="fair-attribute-prediction",
        workspace="tobias-haenel",
        auto_metric_logging=False,
        auto_param_logging=False,
    )
    experiment.set_name(experiment_name)
    log_experiment_status(experiment, "started")
    try:
        experiment.log_parameters(parameters)

        if cudnn.is_available():
            cudnn.enabled = False

        train_dataset = create_dataset(parameters, split_name="train")
        valid_dataset = create_dataset(parameters, split_name="valid")

        train_dataloader = create_dataloader(parameters, train_dataset)
        valid_dataloader = create_dataloader(parameters, valid_dataset)

        model = create_model(parameters, train_dataset)

        optimizer = create_optimizer(parameters, model)
        lr_scheduler = create_lr_scheduler(parameters, optimizer)

        for prediction_attribute_index, prediction_attribute_size in enumerate(model.module.attribute_sizes):
            experiment.log_other(
                f"prediction_attribute_{prediction_attribute_index}_size",
                prediction_attribute_size,
            )
            attribute_index = train_dataset.prediction_attribute_indices[prediction_attribute_index]
            prediction_attribute_class_weights = model.module.attribute_class_weights[prediction_attribute_index]
            for class_index, attribute_class_count in enumerate(train_dataset.attribute_class_counts[attribute_index]):
                experiment.log_other(
                    f"prediction_attribute_{prediction_attribute_index}_class_{class_index}_count",
                    attribute_class_count,
                )
                if prediction_attribute_class_weights is not None:
                    experiment.log_other(
                        f"prediction_attribute_{prediction_attribute_index}_class_{class_index}_weight",
                        prediction_attribute_class_weights[class_index],
                    )

        best_model_state, final_model_state = train_classifier(
            model,
            optimizer,
            lr_scheduler,
            train_dataloader,
            valid_dataloader,
            parameters,
            experiment,
        )

        target_attribute = train_dataset.attribute(parameters["target_attribute_index"])
        final_model_state["scores"], final_model_confusion_matrix = evaluate_classifier(
            model, final_model_state, valid_dataloader, parameters, experiment
        )
        experiment.log_metric("final_accuracy", f"{final_model_state['valid_metrics']['accuracy'] / 100.0:.3}")
        experiment.log_metrics(
            {
                f"final_{score_name}_score": f"{score_value:.3}"
                for score_name, score_value in final_model_state["scores"].items()
            }
        )
        loss_prefix = "additional_loss_fair_"
        experiment.log_metrics(
            {
                f"final_{loss_name[len(loss_prefix):]}_loss": f"{loss_value:.2E}"
                for loss_name, loss_value in final_model_state["valid_metrics"].items()
                if loss_name.startswith(loss_prefix)
            }
        )
        experiment.log_confusion_matrix(
            matrix=final_model_confusion_matrix,
            title="Final Model Confusion Matrix",
            row_label=f"Actual {target_attribute.name}",
            column_label=f"Predicted {target_attribute.name}",
            file_name="final_model_confusion_matrix.json",
        )

        experiment_results_dir_path = Path("experiments") / "results" / experiment_name / start_date.isoformat()
        best_model_state_file_path = experiment_results_dir_path / f"best_model.pt"
        final_model_state_file_path = experiment_results_dir_path / f"final_model.pt"
        parameters_file_path = experiment_results_dir_path / f"parameters.pt"
        experiment_results_dir_path.mkdir(parents=True, exist_ok=True)
        save(best_model_state, best_model_state_file_path)
        save(final_model_state, final_model_state_file_path)
        save(parameters, parameters_file_path)

        experiment.log_model("results", str(experiment_results_dir_path))

        log_experiment_status(experiment, "finished successfully")
    except Exception as exception:
        log_experiment_status(experiment, f"finished with errors\n{traceback.format_exc()}")
        raise exception

    experiment.end()
    return experiment, best_model_state, final_model_state


def run_experiment(args_root_dir_path: Path, relative_args_file_path: Path):
    """
    Performs a fair attribute prediction experiment by reading its hyperparemeters from an argument file (*.args).

    Each argument file may contain the following parameters:
        --batch_size: int = 256
            The batch size for training and evaluation (should be large for good fairness estimates)
        --epoch_count: int = 15
            The number of epochs for training
        --learning_rate: float = 1e-3
            The learning rate for the optimization procedure
        --learning_rate_scheduler: str = "none"
            The learning rate scheduling method ("none" or "reduce_lr_on_plateau") for the optimization procedure
        --reduce_lr_on_plateau_factor: float = 0.5
            ReduceLROnPlateau factor
        --reduce_lr_on_plateau_patience: int = 5
            ReduceLROnPlateau patience
        --dataset: str = "celeba"
            Dataset ("celeba", "siim_isic_melanoma", "utkface")
        --model: str = "slimcnn"
            Model type ("efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4",
                        "efficientnet-b5", "efficientnet-b6", "efficientnet-b7" or "slimcnn")
        --optimizer: str = "adam"
            Optimizer ("adam", "sgd")
        --adam_beta_1: float = 0.9
            Adam optimizer beta_1 parameter
        --adam_beta_2: float = 0.999
            Adam optimizer beta_2 parameter
        --sgd_momentum: float = 0.0
            SGD momentum parameter
        --sensitive_attribute_index: int
            Index of the sensitive attribute
        --target_attribute_index: int
            Index of the target attribute
        --fair_loss_weight: float = 1
            Weighting coefficient for the fair loss term
        --fair_loss_type: str = "intersection_over_union_conditioned"
            Fair loss type ("demographic_parity", "equalized_odds", "intersection_over_union_paired",
                            "intersection_over_union_conditioned", "mutual_information_dp", "mutual_information_eo")
        --pretrained_model: str
            (Optional) path to a pretrained model state file (*.pt) to reuse weights from a previous training procedure
        --class_weights: str = "none"
            Class weighting method for the cross entropy loss ("none", "balanced", "ins", "isns", "ens")
        --ens_beta: float = 0.99
            ENS class weighting beta parameter

    The experiment name is derived from the specified arguments root directory, of its subdirectories
    and of the argument files.

    :param args_root_dir_path: Path pointing to the arguments root directory for this set of experiments
    :param relative_args_file_path: Path of the arguments files relative to the arguments root directory
    """
    absolute_args_file_path = args_root_dir_path / relative_args_file_path

    parser = ArgumentParser(fromfile_prefix_chars="+")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_count", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning_rate_scheduler",
        default="none",
        choices=["none", "reduce_lr_on_plateau"],
    )
    parser.add_argument("--reduce_lr_on_plateau_factor", type=float, default=0.5)
    parser.add_argument("--reduce_lr_on_plateau_patience", type=int, default=5)
    parser.add_argument("--dataset", default="celeba", choices=["utkface", "celeba", "siim_isic_melanoma"])
    parser.add_argument(
        "--model",
        default="slimcnn",
        choices=[
            "slimcnn",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ],
    )
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sensitive_attribute_index", required=True, type=int)
    parser.add_argument("--target_attribute_index", required=True, type=int)
    parser.add_argument("--fair_loss_weight", type=float, default=1)
    parser.add_argument(
        "--fair_loss_type",
        default="intersection_over_union_conditioned",
        choices=fair_losses.keys(),
    )
    parser.add_argument("--pretrained_model")
    parser.add_argument("--class_weights", default="none", choices=["none", "balanced", "ins", "isns", "ens"])
    parser.add_argument("--ens_beta", type=float, default=0.99)

    arguments = parser.parse_args([f"+{absolute_args_file_path}"])

    parameters = deepcopy(vars(arguments))
    experiment_name_parts = [args_root_dir_path.name, *relative_args_file_path.parts[:-1], relative_args_file_path.stem]
    experiment_name = "-".join(experiment_name_parts)

    fair_attribute_prediction_experiment(parameters, experiment_name)
