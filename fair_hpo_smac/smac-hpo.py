#!/usr/bin/env python

import logging
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from os import makedirs, path
from sys import argv

import numpy.random
import torch.random
from numpy.random import RandomState
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario
from torch import cuda, float32, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from data.UTKFace import UTKFaceDataset, load_utkface
from hpo.Hyperparameters import hyperparameters_from_config
from hpo.Cost import cost_functions
from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder

start_date = datetime.now()

arg_parser = ArgumentParser(
    description="Perform HPO with SMAC to train a generative model"
)
dataset_directory_group = arg_parser.add_mutually_exclusive_group(required=True)
dataset_directory_group.add_argument(
    "--utkface-dir",
    help="UTKFace dataset directory",
    required=False,
),
arg_parser.add_argument(
    "--output-dir",
    default=".",
    required=False,
    help="Directory for log files, save states and HPO output",
)
arg_parser.add_argument(
    "--cost",
    default="MS-SSIM",
    choices=["MS-SSIM", "FairMS-SSIM"],
    required=False,
    help="Cost function used for HPO",
)

arg_parser.add_argument(
    "--sensitive-attribute",
    type=int,
    default=0,
    required=False,
    help="Index of the sensitive attribute",
)

arg_parser.add_argument(
    "--image_size",
    default=64,
    type=int,
    required=False,
    help="Image size used for loading the dataset",
)
arg_parser.add_argument(
    "--batch_size",
    default=144,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
arg_parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    required=False,
    help="Epochs used for training the generative model",
)
arg_parser.add_argument(
    "--datasplit-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for creating random train, validation and tast dataset splits",
)
arg_parser.add_argument(
    "--smac-pcs-file",
    required=True,
    help="Parameter configuration file used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-initial-design",
    default="Sobol",
    choices=["DefaultConfiguration", "Sobol"],
    required=False,
    help="Initial design for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-runtime",
    default=604800,
    type=int,
    required=False,
    help="Maximum amount runtime used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-runcount",
    default=None,
    type=int,
    required=False,
    help="Maximum number of runs during hyperparameter optimization with SMAC",
)
args = arg_parser.parse_args(argv[1:])
output_directory = path.join(
    args.output_dir, start_date.strftime("%Y-%m-%d_%H:%M:%S_%f")
)
cost_function_name = args.cost
sensitive_attribute_index = args.sensitive_attribute
image_size = args.image_size
batch_size = args.batch_size
epoch_count = args.epochs
datasplit_seed = args.datasplit_seed
pcs_file = args.smac_pcs_file
max_runtime = args.smac_runtime
max_runcount = args.smac_runcount
smac_seed = args.smac_seed
initial_design_name = args.smac_initial_design

makedirs(output_directory, exist_ok=True)
log_file_path = path.join(output_directory, "log.txt")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, force=True)
print(f"Logging started in output directory {output_directory}")
logging.info(f"Script started at {start_date}")

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
    logging.info(
        f"Memory allocation was selected to be performed on {device_count} "
        f"CUDA device{'s' if device_count > 1 else ''}"
    )
else:
    device = "cpu"
    device_count = 1
    logging.info("Memory allocation was selected to be performed on the CPU device")

if cudnn.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

logging.info(
    f"Data will be loaded with sensitive attribute {sensitive_attribute_index}, "
    f"image size {image_size}, batch size {batch_size} and "
    f"datasplit seed {datasplit_seed}"
)
logging.info(f"Generative model will be trained for {epoch_count} epochs")
logging.info(
    f"Hyperparameter optimisation with SMAC will be run for {max_runtime}s with "
    f"{'' if max_runcount is None else str(max_runcount) + ' evaluations, '}"
    f"parameter configuration file '{pcs_file}', seed {smac_seed} and "
    f"cost function {cost_function_name}"
)

num_workers = device_count * 4
data_state = {
    "sensitive_attribute_index": sensitive_attribute_index,
    "image_size": image_size,
    "batch_size": batch_size,
    "datasplit_seed": datasplit_seed,
}

if args.utkface_dir is not None:
    dataset_directory = args.utkface_dir

    assert 0 <= sensitive_attribute_index < len(UTKFaceDataset.target_attributes)
    sensitive_attribute = UTKFaceDataset.target_attributes[sensitive_attribute_index]

    transform = Compose(
        [ConvertImageDtype(float32), Resize(image_size), Lambda(lambda x: 2 * x - 1)]
    )
    target_transform = Lambda(lambda x: x[sensitive_attribute_index])
    data_state["dataset_directory"] = dataset_directory
    data_state["dataset"] = "UTKFace"
    data_state["sensitive_attribute"] = sensitive_attribute
    train_dataset, validation_dataset, test_dataset = load_utkface(
        random_split_seed=datasplit_seed,
        image_directory_path=dataset_directory,
        transform=transform,
        target_transform=target_transform,
        in_memory=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    logging.info(
        f"UTKFace dataset was loaded from directory '{dataset_directory}' with "
        f"target sensitive attribute {sensitive_attribute.__name__}"
    )
else:
    raise RuntimeError("No dataset was specified")

save_file_directory = path.join(output_directory, "save_states")
makedirs(save_file_directory)
data_save_file_path = path.join(save_file_directory, "data.pt")
save(data_state, data_save_file_path)


def hyperparameter_cost(hyperparameter_config, seed):
    hyperparameter_cost.run += 1

    hyperparameter_cost.seed = seed
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)

    hyperparameters = hyperparameters_from_config(
        hyperparameter_config, max_iteration=epoch_count * len(train_dataloader)
    )
    hyperparameter_cost.hyperparameters = deepcopy(hyperparameters)

    def train_criterion(_model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        train_criterion.iteration += 1
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )

    train_criterion.iteration = 0

    def validation_criterion(_model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )

    model = FlexVAE(
        image_size,
        hyperparameters.latent_dimension_count,
        hyperparameters.hidden_layer_count,
        hyperparameters.vae_loss_gamma,
        hyperparameters.C_max,
        hyperparameters.C_stop_iteration,
        hyperparameters.reconstruction_loss,
        hyperparameters.reconstruction_loss_args,
    )
    if device_count > 1:
        model = DataParallel(model)
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    lr_scheduler = ExponentialLR(optimizer, gamma=hyperparameters.lr_scheduler_gamma)

    train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
        model,
        optimizer,
        lr_scheduler,
        epoch_count,
        train_criterion,
        validation_criterion,
        train_dataloader,
        validation_dataloader,
        schedule_lr_after_epoch=True,
        display_progress=False,
    )

    if isinstance(model, DataParallel):
        model = model.module
    cost, additional_info = hyperparameter_cost.function(
        model, validation_dataloader, sensitive_attribute
    )

    model_state = {
        "run": hyperparameter_cost.run,
        "seed": hyperparameter_cost.seed,
        "hyperparameters": hyperparameter_cost.hyperparameters,
        "cost": cost,
        "additional_info": additional_info,
        "epoch_count": epoch_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_save_file_name = f"model-run-{hyperparameter_cost.run:04}.pt"
    model_save_file_path = path.join(save_file_directory, model_save_file_name)
    save(model_state, model_save_file_path)
    return cost, additional_info


hyperparameter_cost.run = 0
hyperparameter_cost.seed = None
hyperparameter_cost.hyperparameters = None
hyperparameter_cost.function = cost_functions[cost_function_name]


smac_output_directory = path.join(output_directory, "smac")
scenario_dict = {
    "pcs_fn": pcs_file,
    "run_obj": "quality",
    "wallclock-limit": max_runtime,
    "deterministic": "false",
    "limit_resources": False,
    "output_dir": smac_output_directory,
}
if max_runcount is not None:
    scenario_dict["ta_run_limit"] = max_runcount
scenario = Scenario(scenario_dict)
initial_designs = {"DefaultConfiguration": DefaultConfiguration, "Sobol": SobolDesign}
smac = SMAC4HPO(
    scenario=scenario,
    rng=RandomState(smac_seed),
    tae_runner=hyperparameter_cost,
    initial_design=initial_designs[initial_design_name],
    initial_design_kwargs={"n_configs_x_params": 4},
    intensifier_kwargs={"maxR": 5},
)
incumbent_hyperparameter_config = smac.optimize()
run_history = smac.get_runhistory()
trajectory = smac.get_trajectory()
logging.info(f"SMAC HPO finished with Incumbent {incumbent_hyperparameter_config}")

smac_state = {
    "scenario": scenario,
    "seed": smac_seed,
    "cost_function": cost_function_name,
    "incumbent_hyperparameter_config": incumbent_hyperparameter_config,
    "run_history": run_history,
    "trajectory": trajectory,
}
smac_save_file_path = path.join(save_file_directory, "smac.pt")
save(smac_state, smac_save_file_path)

end_date = datetime.now()
duration = end_date - start_date
logging.info(
    f"Script finished at {end_date} with a runtime of {duration.total_seconds()}"
)
