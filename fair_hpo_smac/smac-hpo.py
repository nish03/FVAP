#!/usr/bin/env python

import logging
from argparse import ArgumentParser
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from math import log
from os import makedirs, path
from sys import argv

arg_parser = ArgumentParser(
    description="Perform HPO with SMAC to train a generative model"
)
dataset_directory_group = arg_parser.add_mutually_exclusive_group(required=True)
dataset_directory_group.add_argument(
    "-u",
    "--utkface-dir",
    help="UTKFace dataset directory",
    required=False,
),
arg_parser.add_argument(
    "-o",
    "--output-dir",
    default=".",
    required=False,
    help="Directory for log files, save states and HPO output",
)
arg_parser.add_argument(
    "-c",
    "--cost",
    default="VIFp",
    choices=["VIFp", "FairVIFp"],
    required=False,
    help="Cost function used for HPO",
)

arg_parser.add_argument(
    "-s",
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
    help="Epochs used for training the generative models",
)
arg_parser.add_argument(
    "--datasplit-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for creating random train, validation and tast dataset splits",
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
    default=72000,
    type=int,
    required=False,
    help="Runtime used for hyperparameter optimization with SMAC",
)
args = arg_parser.parse_args(argv[1:])

start_date = datetime.now()
output_directory = path.join(
    args.output_dir, start_date.strftime("%Y-%m-%d_%H:%M:%S_%f")
)
makedirs(output_directory, exist_ok=True)
log_file_path = path.join(output_directory, "log.txt")
# noinspection PyArgumentList
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
print(f"Logging started with Output Directory {output_directory}")

from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from numpy.random import RandomState
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from torch import cuda, float32, save, no_grad, zeros, tensor
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Resize, Lambda
from piq import vif_p

from data.UTKFace import load_utkface, UTKFaceDataset
from models.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder
import torch.random
import numpy.random

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


cost_function_name = args.cost
sensitive_attribute_index = args.sensitive_attribute
image_size = args.image_size
batch_size = args.batch_size
epoch_count = args.epochs
datasplit_seed = args.datasplit_seed
smac_runtime = args.smac_runtime
smac_seed = args.smac_seed


logging.info(
    f"Data will be loaded with sensitive attribute {sensitive_attribute_index}, "
    f"image size {image_size}, batch size {batch_size} and "
    f"datasplit seed {datasplit_seed}"
)
logging.info(f"Generative models will be trained for {epoch_count} epochs")
logging.info(
    f"Hyperparameter optimisation with SMAC will be run for {smac_runtime}s and "
    f"with seed {smac_seed} and cost function {cost_function_name}"
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
        f"UTKFace dataset was loaded from directory {dataset_directory} with "
        f"target sensitive attribute {sensitive_attribute.__name__}"
    )
else:
    raise RuntimeError("No dataset was specified")

save_file_directory = path.join(output_directory, "save_states")
makedirs(save_file_directory)
data_save_file_path = path.join(save_file_directory, "data.pt")
save(data_state, data_save_file_path)


Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        "latent_dimension_count",
        "hidden_layer_count",
        "vae_loss_gamma",
        "C_max",
        "C_stop_iteration",
        "learning_rate",
        "weight_decay",
        "lr_scheduler_gamma",
    ],
)


def vif_p_cost(_model, _dataloader):
    if isinstance(_model, DataParallel):
        _model = _model.module
    _model.eval()

    similarity = zeros(1, dtype=float32)
    processed_data_samples = 0
    with no_grad():
        for data, _ in _dataloader:
            data = data.to(device)
            output = _model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            similarity += vif_p(output, data, reduction="sum").item()
            processed_data_samples += len(data)
    similarity = (similarity / processed_data_samples).item()
    logging.debug(f"  VIFp: {similarity}")
    return -similarity


def fair_vif_p_cost(_model, _dataloader):
    if isinstance(_model, DataParallel):
        _model = _model.module
    _model.eval()

    similarities = zeros(len(sensitive_attribute), dtype=float32)
    processed_data_samples = 0
    with no_grad():
        for data, target in _dataloader:
            data, target = data.to(device), target.to(device)
            output = _model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            similarities += tensor(
                [
                    vif_p(
                        output[target == member.value],
                        data[target == member.value],
                        reduction="sum",
                    )
                    for member in sensitive_attribute
                ]
            )
            processed_data_samples += len(data)

    similarities /= processed_data_samples
    min_similarity = similarities.min().item()
    for member in sensitive_attribute:
        logging.debug(f"  VIFp[{str(member)}]: {similarities[member.value].item()}")
    logging.debug(f"  FairVIFp: {min_similarity}")
    return -min_similarity


cost_functions = {"VIFp": vif_p_cost, "FairVIFp": fair_vif_p_cost}


def hyperparameter_cost(_hyperparameter_config, seed):
    hyperparameter_cost.run += 1

    hyperparameter_cost.seed = seed
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)

    _hyperparameter_config = dict(**_hyperparameter_config)
    hyperparameter_cost.config = deepcopy(_hyperparameter_config)
    hyperparameters = Hyperparameters(**_hyperparameter_config)

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

    model, _, _ = train_variational_autoencoder(
        model,
        optimizer,
        lr_scheduler,
        epoch_count,
        train_criterion,
        validation_criterion,
        train_dataloader,
        validation_dataloader,
        save_model_state,
        schedule_lr_after_epoch=True,
        display_progress=False,
    )
    cost_value = hyperparameter_cost.function(model, validation_dataloader)
    return cost_value


hyperparameter_cost.run = 0
hyperparameter_cost.seed = None
hyperparameter_cost.function = cost_functions[cost_function_name]
hyperparameter_cost.config = {}


def save_model_state(
    epoch, model, optimizer, lr_scheduler, train_epoch_losses, validation_epoch_losses
):
    if isinstance(model, DataParallel):
        model = model.module

    model_state = {
        "run": hyperparameter_cost.run,
        "seed": hyperparameter_cost.seed,
        "epoch": epoch,
        "hyperparameter_config": hyperparameter_cost.config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_save_file_name = f"model-run-{hyperparameter_cost.run:04}.pt"
    model_save_file_path = path.join(save_file_directory, model_save_file_name)
    save(model_state, model_save_file_path)


max_hidden_layer_count = int(log(image_size, 2)) - 1
max_iteration = epoch_count * len(train_dataloader)

hyperparameter_config_space = ConfigurationSpace()

hidden_layer_count_hyperparameter = UniformIntegerHyperparameter(
    "hidden_layer_count",
    1,
    max_hidden_layer_count,
    default_value=1,
)
latent_dimension_count_hyperparameter = UniformIntegerHyperparameter(
    "latent_dimension_count", 16, 512, default_value=128
)
vae_loss_gamma_hyperparameter = UniformFloatHyperparameter(
    "vae_loss_gamma", 1.0, 2000.0, default_value=10.0, log=True
)
C_max_hyperparameter = UniformFloatHyperparameter(
    "C_max", 5.0, 50.0, default_value=25.0
)
C_stop_iteration_hyperparameter = UniformIntegerHyperparameter(
    "C_stop_iteration",
    int(0.05 * max_iteration),
    max_iteration,
    default_value=int(0.2 * max_iteration),
)
learning_rate_hyperparameter = UniformFloatHyperparameter(
    "learning_rate", 5e-6, 5e-3, default_value=5e-4, log=True
)
weight_decay_hyperparameter = UniformFloatHyperparameter(
    "weight_decay", 0.0, 0.25, default_value=0.0
)
lr_scheduler_gamma_hyperparameter = UniformFloatHyperparameter(
    "lr_scheduler_gamma", 0.85, 1.0, default_value=0.95
)
hyperparameter_config_space.add_hyperparameters(
    [
        hidden_layer_count_hyperparameter,
        latent_dimension_count_hyperparameter,
        vae_loss_gamma_hyperparameter,
        C_max_hyperparameter,
        C_stop_iteration_hyperparameter,
        learning_rate_hyperparameter,
        weight_decay_hyperparameter,
        lr_scheduler_gamma_hyperparameter,
    ]
)

smac_output_directory = path.join(output_directory, "smac")
scenario_dict = {
    "run_obj": "quality",
    "wallclock-limit": smac_runtime,
    "cs": hyperparameter_config_space,
    "deterministic": "false",
    "limit_resources": False,
    "output_dir": smac_output_directory,
}
scenario = Scenario(scenario_dict)
smac = SMAC4HPO(
    scenario=scenario,
    rng=RandomState(smac_seed),
    tae_runner=hyperparameter_cost,
    initial_design_kwargs={"init_budget": 20},
)
incumbent_hyperparameter_config = smac.optimize()
logging.info(f"SMAC HPO finished with Incumbent {incumbent_hyperparameter_config}")

smac_state = {
    "scenario": scenario,
    "seed": smac_seed,
    "incumbent_hyperparameter_config": incumbent_hyperparameter_config,
    "cost_function": cost_function_name,
}
smac_save_file_path = path.join(save_file_directory, "smac.pt")
save(smac_state, smac_save_file_path)

end_date = datetime.now()
duration = end_date - start_date
logging.info(
    f"Script finished at {end_date} with a runtime of {duration.total_seconds()}"
)
