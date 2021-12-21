from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype

from CelebA import CelebA
from MultiAttributeDataset import MultiAttributeDataset
from SlimCNN import SlimCNN
from UTKFace import UTKFace


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def create_dataset(parameters: dict, split_name: str):
    if parameters["dataset"] == "UTKFace":
        return UTKFace(
            dataset_dir_path=Path("datasets") / "UTKFace",
            image_transform=ConvertImageDtype(torch.float32),
            split_name=split_name,
        )
    elif parameters["dataset"] == "CelebA":
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
    if parameters["model_name"] == "SlimCNN":
        model = SlimCNN(
            attribute_sizes=train_dataset.attribute_sizes,
        )
        model = DataParallel(model)
        model.to(get_device())
        return model


def create_optimizer(parameters: dict, model: torch.nn.Module):
    if parameters["optimizer"] == "Adam":
        return Adam(model.parameters(), lr=parameters["learning_rate"])
    return None
