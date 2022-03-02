from pathlib import Path

import torch
from torch import Generator, randperm, tensor, arange
from torchvision.io import read_image

from multi_attribute_dataset import MultiAttributeDataset


class SIIMISICMelanoma(MultiAttributeDataset):
    def __init__(
        self,
        dataset_dir_path,
        image_transform=None,
        attribute_transform=None,
        split_name="all",
        split_seed=42,
        split_valid_test_ratio=0.5,
    ):
        MultiAttributeDataset.__init__(self, ["sex", "age", "diagnosis"], [2, 5, 2])
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {dataset_dir_path} - does not exist")
        self.split_name = split_name
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {dataset_dir_path} - does not exist")
        train_image_directory_path = dataset_dir_path / "jpeg" / "train"
        test_image_directory_path = dataset_dir_path / "jpeg" / "test"
        train_image_file_paths = sorted(train_image_directory_path.glob("*.jpg"))
        test_image_file_paths = sorted(test_image_directory_path.glob("*.jpg"))
        self.image_file_paths = train_image_file_paths + test_image_file_paths
        image_range = range(len(self.image_file_paths))
        train_image_range = range(len(train_image_file_paths))
        valid_image_range = range(
            train_image_range.stop,
            train_image_range.stop + int(split_valid_test_ratio * len(test_image_file_paths)),
        )
        test_image_range = range(valid_image_range.stop, image_range.stop)
        split_image_ranges = {
            "all": image_range,
            "train": train_image_range,
            "valid": valid_image_range,
            "test": test_image_range,
        }
        self.image_file_indices = arange(len(self.image_file_paths))[split_image_ranges[self.split_name]]
        self.image_file_indices = self.image_file_indices[
            randperm(len(self.image_file_indices), generator=Generator().manual_seed(split_seed))
        ]
        self.attribute_data = []

    def __len__(self) -> int:
        return len(self.image_file_indices)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        image_file_index = self.image_file_indices[index]
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        attribute_values = tensor([1, 2, 0])
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values
