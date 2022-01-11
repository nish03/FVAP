from datetime import datetime
from pathlib import Path
from typing import Dict

from comet_ml import OfflineExperiment, Experiment
from torch import save

from training import train_classifier
from util import create_dataset, create_dataloader, create_model, create_optimizer


def train_model_experiment(parameters: Dict, experiment_name: str, offline_experiment: bool = False):
    start_date = datetime.utcnow()
    experiment_class = OfflineExperiment if offline_experiment else Experiment
    experiment = experiment_class(
        project_name="fair-attribute-prediction",
        workspace="tobias-haenel",
        auto_metric_logging=False,
        auto_param_logging=False,
    )
    print("Started train model experiment")
    try:

        experiment.set_name(experiment_name)
        experiment.log_parameters(parameters)

        train_dataset = create_dataset(parameters, split_name="train")
        valid_dataset = create_dataset(parameters, split_name="valid")

        train_dataloader = create_dataloader(parameters, train_dataset)
        valid_dataloader = create_dataloader(parameters, valid_dataset)

        model = create_model(parameters, train_dataset)
        optimizer = create_optimizer(parameters, model)

        best_model_state, final_model_state = train_classifier(
            model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            parameters,
            experiment,
        )

        experiment_dir_path = Path("experiments") / experiment_name / start_date.isoformat()
        best_model_checkpoint_file_path = experiment_dir_path / f"best_model.pt"
        final_model_checkpoint_file_path = experiment_dir_path / f"final_model.pt"
        parameters_file_path = experiment_dir_path / f"parameters.pt"
        experiment_dir_path.mkdir(parents=True, exist_ok=True)
        save(best_model_state, best_model_checkpoint_file_path)
        save(final_model_state, final_model_checkpoint_file_path)
        save(parameters, parameters_file_path)

        experiment.log_model("results", str(experiment_dir_path))
    except Exception as exception:
        raise exception
    finally:
        experiment.end()
        print("Finished train model experiment")
    return experiment
