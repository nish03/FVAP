from csv import reader
from pathlib import Path

from torch import arange, tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from numpy import loadtxt
import numpy


class CelebA(Dataset):
    def __init__(
        self,
        dataset_dir_path,
        split_name="all",
        image_transform=None,
        attribute_transform=None,
    ):
        dataset_dir_path = Path(dataset_dir_path)
        if not dataset_dir_path.is_dir():
            raise ValueError(
                f"Invalid dataset directory path {dataset_dir_path} - does not exist"
            )
        self.image_dir_path = dataset_dir_path / "img_align_celeba"
        self.partitions_file_path = dataset_dir_path / "list_eval_partition.txt"
        self.attribute_data_file_path = dataset_dir_path / "list_attr_celeba.txt"

        self.attribute_data = loadtxt(
            "datasets/CelebA/celeba/list_attr_celeba.txt",
            dtype=numpy.int64,
            skiprows=2,
            usecols=range(1, 41),
        )
        with open(self.attribute_data_file_path, "r") as attribute_data_file:
            attribute_data_reader = reader(attribute_data_file, delimiter=" ")

            self.dataset_image_count = int(next(attribute_data_reader)[0])
            self.attribute_names = next(attribute_data_reader)[:-1]

        self.image_file_paths = []
        self.split_image_indices = None
        self.image_file_numbers, self.partition_indices = loadtxt(
            str(self.partitions_file_path),
            dtype=numpy.int64,
            converters={0: lambda image_file_name: int(image_file_name.split(b".")[0])},
            unpack=True,
        )
        self.attribute_data = loadtxt(
            str(self.attribute_data_file_path),
            dtype=numpy.int64,
            skiprows=2,
            usecols=range(1, 41),
        )
        self.attribute_data[self.attribute_data == -1] = 0
        self.split(split_name)
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform

    def __len__(self):
        return self.split_image_indices.shape[0]

    def __getitem__(self, index):
        image_index = self.split_image_indices[index]
        image_file_path = (
            self.image_dir_path / f"{self.image_file_numbers[image_index]:0>6}.jpg"
        )
        image = read_image(str(image_file_path))
        attribute_values = tensor(self.attribute_data[image_index])
        if self.image_transform:
            image = self.image_transform(attribute_values)
        if self.attribute_transform:
            attribute_values = self.attribute_transform(attribute_values)
        return image, attribute_values

    def split(self, split_name):
        split_name_to_partition_index = {"train": 0, "valid": 1, "test": 2}
        self.split_image_indices = arange(self.dataset_image_count)
        if split_name in split_name_to_partition_index:
            self.split_image_indices = self.split_image_indices[
                self.partition_indices == split_name_to_partition_index[split_name]
            ]
