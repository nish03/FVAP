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
    status_message = f"Experiment '{experiment.get_name()}' {status}"
    print(status_message)
    experiment.send_notification(status_message)


def fair_attribute_prediction_experiment(parameters: Dict, experiment_name: str):
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
        best_model_state["scores"], best_model_confusion_matrix = evaluate_classifier(
            model, best_model_state, valid_dataloader, parameters, experiment
        )
        experiment.log_metrics(
            {f"best_model_{score_name}": score_value for score_name, score_value in best_model_state["scores"].items()}
        )
        experiment.log_confusion_matrix(
            matrix=best_model_confusion_matrix,
            title="Best Model Confusion Matrix",
            row_label=f"Actual {target_attribute.name}",
            column_label=f"Predicted {target_attribute.name}",
            file_name="best_model_confusion_matrix.json"
        )
        final_model_state["scores"], final_model_confusion_matrix = evaluate_classifier(
            model, final_model_state, valid_dataloader, parameters, experiment
        )
        experiment.log_metrics(
            {
                f"final_model_{score_name}": score_value
                for score_name, score_value in final_model_state["scores"].items()
            }
        )
        experiment.log_confusion_matrix(
            matrix=final_model_confusion_matrix,
            title="Final Model Confusion Matrix",
            row_label=f"Actual {target_attribute.name}",
            column_label=f"Predicted {target_attribute.name}",
            file_name="final_model_confusion_matrix.json"
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
    return experiment


def run_experiment(args_root_dir_path: Path, relative_args_file_path: Path):
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
    parser.add_argument("--model", default="slimcnn", choices=["slimcnn", "simplecnn"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sensitive_attribute_index", required=True, type=int)
    parser.add_argument("--target_attribute_index", required=True, type=int)
    parser.add_argument("--fair_loss_weight", type=float, default=1)
    parser.add_argument(
        "--fair_loss_type",
        default="intersection_over_union",
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
