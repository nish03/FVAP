from pathlib import Path

import torch
from torch import Generator, randperm, tensor
from torchvision.io import read_image
from pandas import read_csv

from multi_attribute_dataset import MultiAttributeDataset


class SIIMISICMelanoma(MultiAttributeDataset):
    def __init__(
        self,
        dataset_dir_path,
        image_transform=None,
        attribute_transform=None,
        split_name="all",
        split_seed=42,
        split_train_factor=0.7,
        split_valid_factor=0.2,
    ):
        MultiAttributeDataset.__init__(self, ["age", "gender", "diagnosis"], [5, 2, 2])
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {dataset_dir_path} - does not exist")
        image_directory_path = dataset_dir_path / "jpeg" / "train"
        data_file_path = dataset_dir_path / "train.csv"
        self.image_file_paths = sorted(image_directory_path.glob("*.jpg"))
        self.attribute_data = read_csv(data_file_path).sort_values(by="image_name")
        self.attribute_data['sex'].fillna(self.attribute_data['sex'].mode()[0], inplace=True)
        self.attribute_data['age_approx'].fillna(self.attribute_data['age_approx'].mode()[0], inplace=True)
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        self.split_name = split_name
        self.image_file_indices = randperm(len(self.image_file_paths), generator=Generator().manual_seed(split_seed))
        image_range = range(len(self.image_file_indices))
        train_image_range = range(int(split_train_factor * len(image_range)))
        valid_image_range = range(
            train_image_range.stop,
            train_image_range.stop + int(split_valid_factor * len(image_range)),
        )
        test_image_range = range(valid_image_range.stop, image_range.stop)
        split_image_ranges = {
            "all": image_range,
            "train": train_image_range,
            "valid": valid_image_range,
            "test": test_image_range,
        }
        self.image_file_indices = self.image_file_indices[split_image_ranges[self.split_name]]

    def __len__(self) -> int:
        return len(self.image_file_indices)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        image_file_index = self.image_file_indices[index].item()
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        sex, age_approx, target = self.attribute_data.iloc[image_file_index, [2, 3, 7]]
        diagnosis = int(target)
        if sex == "male":
            gender = 0
        elif sex == "female":
            gender = 1
        else:
            raise ValueError(f"Invalid sex {sex}")
        if age_approx <= 20:
            age = 0
        elif 21 <= age_approx <= 40:
            age = 1
        elif 41 <= age_approx <= 60:
            age = 2
        elif 61 <= age_approx <= 80:
            age = 3
        elif 81 <= age_approx:
            age = 4
        else:
            raise ValueError(f"Invalid age {age_approx}")

        attribute_values = tensor([age, gender, diagnosis])
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values
