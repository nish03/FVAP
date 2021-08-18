#!/usr/bin/env python

import logging
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from os import makedirs, path
from sys import argv

arg_parser = ArgumentParser(
    description="Replication study of PyTorch-VAE github repository "
    "(2020 - Subramian, A.K. - https://github.com/AntixK/PyTorch-VAE)"
)
arg_parser.add_argument(
    "-c",
    "--celeba-dir",
    default=".",
    required=False,
    help="CelebA dataset directory",
)
arg_parser.add_argument(
    "-o",
    "--output-dir",
    default=".",
    required=False,
    help="Directory for log files, save states and SMAC output",
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

from torch import cuda, save
import torch.random
import numpy.random
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    Resize,
    ToTensor,
    Lambda,
)
from torchvision.datasets.celeba import CelebA

from models.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder

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

image_size = 64
batch_size = 144
epoch_count = 50
random_seed = 1265
torch.random.manual_seed(random_seed)
numpy.random.seed(random_seed)

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

hyperparameters = Hyperparameters(
    latent_dimension_count=128,
    hidden_layer_count=5,
    vae_loss_gamma=10,
    C_max=25,
    C_stop_iteration=10000,
    learning_rate=0.0005,
    weight_decay=0.0,
    lr_scheduler_gamma=0.95,
)

logging.info(
    f"Data will be loaded with image size {image_size} and batch size {batch_size}"
)
logging.info(f"Variational Autoencoder will be trained for {epoch_count} epochs")

num_workers = device_count * 4

data_state = {
    "image_size": image_size,
    "batch_size": batch_size,
}

dataset_directory = args.celeba_dir
makedirs(dataset_directory, exist_ok=True)

transform = Compose(
    [
        RandomHorizontalFlip(),
        CenterCrop(148),
        Resize(image_size),
        ToTensor(),
        Lambda(lambda x: 2.0 * x - 1.0),
    ]
)
data_state["dataset_directory"] = dataset_directory

train_dataset, validation_dataset = [
    CelebA(root=dataset_directory, split=split, transform=transform, download=False)
    for split in ["train", "valid"]
]
train_dataloader, validation_dataloader = [
    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    for dataset in [train_dataset, validation_dataset]
]

logging.info(f"CelebA dataset was loaded from directory {dataset_directory}")

save_file_directory = path.join(output_directory, "save_states")
makedirs(save_file_directory)
data_save_file_path = path.join(save_file_directory, "data.pt")
save(data_state, data_save_file_path)


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


def save_model_state(
    epoch,
    _model,
    _optimizer,
    _lr_scheduler,
    train_epoch_losses,
    validation_epoch_losses,
):
    is_save_epoch = epoch % (epoch_count // 5) == 0

    if not is_save_epoch:
        return

    model_state = {
        "epoch": epoch,
        "model_state_dict": _model.module.state_dict()
        if isinstance(_model, DataParallel)
        else _model.state_dict(),
        "optimizer_state_dict": _optimizer.state_dict(),
        "lr_scheduler_state_dict": _lr_scheduler.state_dict(),
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_save_file_name = f"model-epoch-{epoch:04}.pt"
    model_save_file_path = path.join(save_file_directory, model_save_file_name)
    save(model_state, model_save_file_path)


logging.debug(
    f"Variational autoencoder training started with {hyperparameters}"
    f"for {epoch_count} epochs"
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

train_variational_autoencoder(
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

end_date = datetime.now()
duration = end_date - start_date
logging.info(
    f"Script finished at {end_date} with a runtime of {duration.total_seconds()}"
)
