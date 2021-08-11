#!/usr/bin/env python

import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from datetime import datetime
from math import log
from os import environ, mkdir, path
from os.path import isdir

from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from numpy.random import RandomState
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from torch import cuda, float32, save, tensor
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from data.UTKFace import UTKFaceDataset, load_utkface
from evaluation.Evaluation import evaluate_variational_autoencoder
from models.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder

current_date = datetime.now()
output_directory = path.join(
    environ["HOME"],
    "data",
    "discoret",
    f"fair-hpo-experiments_{current_date.strftime('%Y-%m-%d_%H:%M:%S_%f')}",
)
if not isdir(output_directory):
    mkdir(output_directory)
log_file_path = path.join(output_directory, "log.txt")
# noinspection PyArgumentList
logging.basicConfig(
    filename=log_file_path, encoding="utf-8", level=logging.DEBUG, force=True
)
logging.info(f"Script started at {current_date}")
print(f"Logging started with Output Directory ({output_directory})")

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
    cudnn.benchmark = True
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

dataset_directory = path.join(environ["HOME"], "data", "discoret", "UTKFace")
image_size = 64

transforms = [ConvertImageDtype(float32), Resize(image_size)]
complete_dataset = UTKFaceDataset(dataset_directory, transform=Compose(transforms))
complete_dataloader = DataLoader(complete_dataset, batch_size=len(complete_dataset))
images, _ = next(iter(complete_dataloader))
color_mean = images.mean((0, 2, 3))
color_std = images.std((0, 2, 3))
del images
normalize_color = Normalize(mean=color_mean, std=color_std)
denormalize_color = Normalize(mean=-color_mean / color_std, std=1 / color_std)
dataset_transform = Compose(transforms)
train_dataset, validation_dataset, test_dataset = load_utkface(
    image_directory_path=dataset_directory,
    transform=Compose(transforms),
    target_transform=lambda attributes: tensor([*attributes]),
    in_memory=True,
)
logging.info(
    f"Datasets were loaded with Directory({dataset_directory}), "
    f"Image Size({image_size}), Normalization Color Mean({color_mean}) and "
    f"Normalization Color Standard Deviation({color_std})"
)

batch_size = 144
num_workers = device_count * 4

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
logging.info(f"Data Loaders were created with Batch Size({batch_size})")

epoch_count = 4
logging.info(f"Generative Models will be trained with Epochs({epoch_count})")


def train_generative_model(
    hyperparameters,
):
    def train_criterion(data, _, output, mu, log_var, data_fraction):
        train_criterion.iteration += 1
        return FlexVAE.criterion(
            data,
            output,
            mu,
            log_var,
            hyperparameters.vae_loss_gamma,
            hyperparameters.C_max,
            hyperparameters.C_stop_iteration,
            train_criterion.iteration,
            data_fraction,
        )

    train_criterion.iteration = 0

    def validation_criterion(data, _, output, mu, log_var, data_fraction):
        return FlexVAE.criterion(
            data,
            output,
            mu,
            log_var,
            hyperparameters.vae_loss_gamma,
            hyperparameters.C_max,
            hyperparameters.C_stop_iteration,
            train_criterion.iteration,
            data_fraction,
        )

    model = FlexVAE(
        image_size,
        hyperparameters.latent_dimension_count,
        hyperparameters.hidden_layer_count,
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

    train_epoch_losses = defaultdict(list)
    validation_epoch_losses = defaultdict(list)

    logging.debug(
        f"Generative model training started with {hyperparameters}"
        f"and Epochs({epoch_count})"
    )
    for epoch in range(1, epoch_count + 1):
        logging.debug(f"  Epoch: {epoch}")

        train_losses = train_variational_autoencoder(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            train_criterion,
            display_progress=False,
        )
        logging.debug(
            "    Training Losses - "
            + " ".join([f"{name}: {value}" for name, value in train_losses.items()])
        )

        validation_losses = evaluate_variational_autoencoder(
            model, validation_dataloader, validation_criterion
        )
        logging.debug(
            "    Validation Losses - "
            + " ".join(
                [f"{name}: {value}" for name, value in validation_losses.items()]
            )
        )

        for name, value in train_losses.items():
            train_epoch_losses[name].append(value)
        for name, value in validation_losses.items():
            validation_epoch_losses[name].append(value)

        try_save_state(
            epoch,
            model,
            optimizer,
            lr_scheduler,
            train_epoch_losses,
            validation_epoch_losses,
        )

    return model, train_epoch_losses, validation_epoch_losses


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


def hyperparameter_cost(hyperparameter_config):
    hyperparameter_cost.run += 1

    hyperparameter_config = dict(**hyperparameter_config)
    hyperparameter_cost.current_config = deepcopy(hyperparameter_config)
    hyperparameter_config["C_stop_iteration"] = int(
        hyperparameter_config["C_stop_iteration_fraction"] * max_iteration
    )
    del hyperparameter_config["C_stop_iteration_fraction"]
    hyperparameters = Hyperparameters(**hyperparameter_config)

    model, train_epoch_losses, validation_epoch_losses = train_generative_model(
        hyperparameters
    )
    final_cost = min(validation_epoch_losses[hyperparameter_cost.loss])
    return final_cost


hyperparameter_cost.best = float("inf")
hyperparameter_cost.run = 0
hyperparameter_cost.loss = "Reconstruction"
hyperparameter_cost.current_config = {}


def try_save_state(
    epoch, model, optimizer, lr_scheduler, train_epoch_losses, validation_epoch_losses
):
    cost = validation_epoch_losses[hyperparameter_cost.loss][-1]
    is_best = hyperparameter_cost.best > cost
    if is_best:
        hyperparameter_cost.best = cost

    is_save_epoch = epoch % (epoch_count // 2) == 0

    if not is_best and not is_save_epoch:
        return

    save_state = {
        "run": hyperparameter_cost.run,
        "epoch": epoch,
        "hyperparameter_config": hyperparameter_cost.current_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    if is_best:
        save_file_name = "best.pt"
        save_file_path = path.join(try_save_state.save_file_directory, save_file_name)
        save(save_state, save_file_path)

    if is_save_epoch:
        save_file_name = f"run-{hyperparameter_cost.run:04}_epoch-{epoch:04}.pt"
        save_file_path = path.join(try_save_state.save_file_directory, save_file_name)
        save(save_state, save_file_path)


try_save_state.save_file_directory = path.join(output_directory, "save_states")
if not isdir(try_save_state.save_file_directory):
    mkdir(try_save_state.save_file_directory)

max_hidden_layer_count = int(log(image_size, 2)) - 1
max_iteration = epoch_count * len(train_dataloader)

hyperparameter_config_space = ConfigurationSpace()

hidden_layer_count_hyperparameter = UniformIntegerHyperparameter(
    "hidden_layer_count",
    1,
    max_hidden_layer_count,
    default_value=max_hidden_layer_count,
)
latent_dimension_count_hyperparameter = UniformIntegerHyperparameter(
    "latent_dimension_count", 16, 512, default_value=128
)
vae_loss_gamma_hyperparameter = UniformFloatHyperparameter(
    "vae_loss_gamma", 1.0, 10000.0, default_value=10.0, log=True
)
C_max_hyperparameter = UniformFloatHyperparameter(
    "C_max", 0.0, 50.0, default_value=25.0
)
C_stop_iteration_fraction_hyperparameter = UniformFloatHyperparameter(
    "C_stop_iteration_fraction", 0.1, 1.0, default_value=0.5
)
learning_rate_hyperparameter = UniformFloatHyperparameter(
    "learning_rate", 1e-5, 1e-2, default_value=5e-4, log=True
)
weight_decay_hyperparameter = UniformFloatHyperparameter(
    "weight_decay", 0.0, 0.2, default_value=0.0
)
lr_scheduler_gamma_hyperparameter = UniformFloatHyperparameter(
    "lr_scheduler_gamma", 0.9, 1.0, default_value=0.95
)
hyperparameter_config_space.add_hyperparameters(
    [
        hidden_layer_count_hyperparameter,
        latent_dimension_count_hyperparameter,
        vae_loss_gamma_hyperparameter,
        C_max_hyperparameter,
        C_stop_iteration_fraction_hyperparameter,
        learning_rate_hyperparameter,
        weight_decay_hyperparameter,
        lr_scheduler_gamma_hyperparameter,
    ]
)

smac_output_directory = path.join(output_directory, "smac")
time_limit = 30 * 60
seed = 42
scenario = {
    "run_obj": "quality",
    "runcount-limit": 4,
    "cs": hyperparameter_config_space,
    "deterministic": "false",
    "limit_resources": False,
    "output_dir": smac_output_directory,
}
logging.info(f"SMAC HPO started with Scenario({scenario}) and Seed({seed})")
smac = SMAC4HPO(
    scenario=Scenario(scenario), rng=RandomState(seed), tae_runner=hyperparameter_cost
)
incumbent_hyperparameter_config = smac.optimize()
logging.info(f"SMAC HPO finished with Incumbent {incumbent_hyperparameter_config}")

logging.info(f"Script finished at {datetime.now()}")
