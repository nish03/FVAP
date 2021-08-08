from glob import glob
from os.path import basename as path_basename
from os.path import join as path_join

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import read_image


class UTKFaceDataset(Dataset):
    gender_categories = ["Male", "Female"]
    race_categories = ["White", "Black", "Asian", "Indian", "Other"]

    def __init__(
        self,
        image_directory_path,
        transform=None,
        target_transform=None,
    ):
        self.image_directory = image_directory_path
        self.image_file_paths = glob(path_join(image_directory_path, "*_*_*_*.jpg"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, index):
        image_file_path = self.image_file_paths[index]
        image_data = read_image(image_file_path)
        if self.transform:
            image_data = self.transform(image_data)
        image_file_name = path_basename(image_file_path)
        image_file_name_sections = image_file_name.split("_")
        age, gender, race = [int(x) for x in image_file_name_sections[0:3]]
        target = (age, gender, race)
        if self.target_transform:
            target = self.target_transform(target)
        return image_data, target


def load_utkface(train_split_factor=0.7, validation_split_factor=0.2, **kwargs):
    dataset = UTKFaceDataset(
        **kwargs,
    )
    train_count = int(train_split_factor * len(dataset))
    validation_count = int(validation_split_factor * len(dataset))
    test_count = len(dataset) - train_count - validation_count
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_count, validation_count, test_count]
    )
    return train_dataset, validation_dataset, test_dataset
