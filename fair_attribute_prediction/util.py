from pathlib import Path

import torch
from torch import load
from torch.nn import DataParallel
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype

from celeba import CelebA
from multi_attribute_dataset import MultiAttributeDataset
from simplecnn import SimpleCNN
from slimcnn import SlimCNN
from utkface import UTKFace


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def create_dataset(parameters: dict, split_name: str):
    if parameters["dataset"] == "utkface":
        return UTKFace(
            dataset_dir_path=Path("datasets") / "UTKFace",
            image_transform=ConvertImageDtype(torch.float32),
            split_name=split_name,
        )
    elif parameters["dataset"] == "celeba":
        return CelebA(
            dataset_dir_path=Path("datasets") / "CelebA" / "celeba",
            image_transform=ConvertImageDtype(torch.float32),
            split_name=split_name,
        )
    return None


def create_dataloader(parameters: dict, dataset: torch.utils.data.Dataset):
    num_workers = min(4 * get_device_count(), 8)
    dataloader = DataLoader(
        dataset,
        batch_size=parameters["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def create_model(parameters: dict, train_dataset: MultiAttributeDataset):
    if parameters["model"] == "slimcnn":
        model = SlimCNN(
            attribute_sizes=train_dataset.attribute_sizes,
        )
    elif parameters["model"] == "simplecnn":
        model = SimpleCNN(
            attribute_sizes=train_dataset.attribute_sizes,
        )
    else:
        return None
    model = DataParallel(model)
    model.to(get_device())

    if "pretrained_model" in parameters and parameters["pretrained_model"] is not None:
        pretrained_model_state_file_path = Path(parameters["pretrained_model"])
        if not pretrained_model_state_file_path.is_file():
            raise FileNotFoundError(
                f"Pretrained model state file {str(pretrained_model_state_file_path)} doesn't exist"
            )
        pretrained_model_state = load(pretrained_model_state_file_path)
        pretrained_model_state_dict = pretrained_model_state["model_state_dict"]
        model.load_state_dict(pretrained_model_state_dict)

    return model


def create_optimizer(parameters: dict, model: torch.nn.Module):
    if parameters["optimizer"] == "adam":
        return Adam(
            model.parameters(),
            lr=parameters["learning_rate"],
            betas=(parameters["adam_beta_1"], parameters["adam_beta_2"]),
        )
    if parameters["optimizer"] == "sgd":
        return SGD(model.parameters(), lr=parameters["learning_rate"], momentum=parameters["sgd_momentum"])
    return None


def create_lr_scheduler(parameters: dict, optimizer: torch.optim.Optimizer):
    if parameters["learning_rate_scheduler"] == "reduce_lr_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            patience=parameters["reduce_lr_on_plateau_patience"],
            factor=parameters["reduce_lr_on_plateau_factor"],
            verbose=True,
        )
    return None


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
