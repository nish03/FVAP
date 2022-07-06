from pathlib import Path

import torch
from torch import Generator, randperm, tensor
from torchvision.io import read_image
from pandas import read_csv, DataFrame

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
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {dataset_dir_path} - does not exist")
        image_directory_path = dataset_dir_path / "jpeg" / "train"
        data_file_path = dataset_dir_path / "train.csv"
        self.image_file_paths = sorted(image_directory_path.glob("*.jpg"))
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
        self.image_file_indices = self.image_file_indices[split_image_ranges[self.split_name]].tolist()
        raw_attribute_data = read_csv(data_file_path).sort_values(by="image_name")
        raw_attribute_data["sex"].fillna(raw_attribute_data["sex"].mode()[0], inplace=True)
        raw_attribute_data["age_approx"].fillna(raw_attribute_data["age_approx"].mode()[0], inplace=True)
        attribute_names = ["age", "gender", "diagnosis"]
        self.attribute_data = {attribute_name: [] for attribute_name in attribute_names}
        for image_file_index in self.image_file_indices:
            sex, age_approx, target = raw_attribute_data.iloc[image_file_index, [2, 3, 7]]
            if age_approx <= 30:
                age = 0
            elif 31 <= age_approx <= 60:
                age = 1
            else:
                age = 2
            if sex == "male":
                gender = 0
            else:
                gender = 1
            diagnosis = int(target)
            self.attribute_data[attribute_names[0]].append(age)
            self.attribute_data[attribute_names[1]].append(gender)
            self.attribute_data[attribute_names[2]].append(diagnosis)
        self.attribute_data = DataFrame(self.attribute_data)
        attribute_sizes = [3, 2, 2]
        attribute_class_counts = [
            self.attribute_data[attribute_name].value_counts().tolist() for attribute_name in attribute_names
        ]
        prediction_attribute_indices = [2]
        MultiAttributeDataset.__init__(
            self, attribute_names, attribute_sizes, attribute_class_counts, prediction_attribute_indices
        )

    def __len__(self) -> int:
        return len(self.image_file_indices)

    def _get_sample(self, index: int) -> (torch.Tensor, torch.Tensor):
        image_file_index = self.image_file_indices[index]
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        age, gender, diagnosis = self.attribute_data.iloc[index]
        attribute_values = tensor([age, gender, diagnosis])
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values
