from csv import reader
from pathlib import Path

import numpy
import torch

from multi_attribute_dataset import MultiAttributeDataset
from numpy import arange, loadtxt, unique
from torch import tensor
from torchvision.io import read_image


class CelebA(MultiAttributeDataset):
    def __init__(
        self,
        dataset_dir_path,
        image_transform=None,
        attribute_transform=None,
        split_name="all",
    ):
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {self.dataset_dir_path} - " f"does not exist")
        self.image_dir_path = self.dataset_dir_path / "img_align_celeba"
        self.partitions_file_path = self.dataset_dir_path / "list_eval_partition.txt"
        self.attribute_data_file_path = self.dataset_dir_path / "list_attr_celeba.txt"

        with open(self.attribute_data_file_path, "r") as attribute_data_file:
            attribute_data_reader = reader(attribute_data_file, delimiter=" ")

            self.dataset_image_count = int(next(attribute_data_reader)[0])
            attribute_names = next(attribute_data_reader)[:-1]
        attribute_count = len(attribute_names)

        self.image_file_numbers, partition_indices = loadtxt(
            str(self.partitions_file_path),
            dtype=numpy.int64,
            converters={0: lambda image_file_name: int(image_file_name.split(b".")[0])},
            unpack=True,
        )
        self.attribute_data = loadtxt(
            str(self.attribute_data_file_path),
            dtype=numpy.int64,
            skiprows=2,
            usecols=range(1, attribute_count + 1),
        )
        self.split_name = split_name
        if self.split_name == "all":
            self.partition_indices = partition_indices
        else:
            image_indices = self._split_image_indices(partition_indices, split_name)
            self.attribute_data = self.attribute_data[image_indices]
            self.image_file_numbers = self.image_file_numbers[image_indices]
        self.attribute_data[self.attribute_data == -1] = 0
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        attribute_sizes = [2] * attribute_count
        attribute_class_counts = [
            unique(self.attribute_data[:, attribute_index], return_counts=True)[1].tolist()
            for attribute_index in range(attribute_count)
        ]
        prediction_attribute_indices = list(range(attribute_count))
        MultiAttributeDataset.__init__(
            self, attribute_names, attribute_sizes, attribute_class_counts, prediction_attribute_indices
        )

    def __len__(self):
        return self.attribute_data.shape[0]

    def _get_sample(self, image_index: int) -> (torch.Tensor, torch.Tensor):
        image_file_path = self.image_dir_path / f"{self.image_file_numbers[image_index]:0>6}.jpg"
        image = read_image(str(image_file_path))
        attribute_values = tensor(self.attribute_data[image_index])
        if self.image_transform:
            image = self.image_transform(image)
        if self.attribute_transform:
            attribute_values = self.attribute_transform(attribute_values)
        return image, attribute_values

    def _split_image_indices(self, partition_indices, split_name):
        split_name_to_partition_index = {"train": 0, "valid": 1, "test": 2}
        split_image_indices = arange(self.dataset_image_count)
        if split_name in split_name_to_partition_index:
            split_image_indices = split_image_indices[partition_indices == split_name_to_partition_index[split_name]]
        return split_image_indices
