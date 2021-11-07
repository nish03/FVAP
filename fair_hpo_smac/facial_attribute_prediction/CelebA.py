from csv import reader
from itertools import islice
from pathlib import Path

from torch import tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


class CelebA(Dataset):
    def __init__(
        self,
        dataset_dir_path,
        split_name="all",
        image_transform=None,
        attribute_transform=None,
        in_memory=False,
    ):
        dataset_dir_path = Path(dataset_dir_path)
        if not dataset_dir_path.is_dir():
            raise ValueError(
                f"Invalid dataset directory path {dataset_dir_path} - does not exist"
            )
        self.image_dir_path = dataset_dir_path / "img_align_celeba"
        self.partitions_file_path = dataset_dir_path / "list_eval_partition.txt"
        self.attribute_data_file_path = dataset_dir_path / "list_attr_celeba.txt"
        self.partition_indices = []
        self.image_file_paths = []
        self.attribute_data = []
        with open(self.partitions_file_path, "r") as partitions_file:
            partitions_reader = reader(partitions_file, delimiter=" ")
            for (image_file_name, partition_index) in partitions_reader:
                self.partition_indices.append(int(partition_index))
                self.image_file_paths.append(self.image_dir_path / image_file_name)
        with open(self.attribute_data_file_path, "r") as attribute_data_file:
            attribute_data_reader = reader(attribute_data_file, delimiter=" ")
            for attribute_row in islice(attribute_data_reader, 2, None):
                attribute_row = [
                    attribute_value
                    for attribute_value in attribute_row
                    if attribute_value != ""
                ]
                self.attribute_data.append(
                    [
                        1 if attribute_row[i] == "1" else 0
                        for i in range(1, len(attribute_row))
                    ]
                )

        self.image_transform = image_transform
        self.attribute_transform = attribute_transform

        self.split(split_name)

        self.image_data = []
        self.in_memory = in_memory
        if self.in_memory:
            self.image_data = [
                self.load_image(i) for i in range(len(self.image_file_paths))
            ]

    def __len__(self):
        return len(self.image_file_paths)

    def load_image(self, index):
        image = read_image(str(self.image_file_paths[index]))
        if self.image_transform:
            image = self.image_transform(image)
        return image

    def __getitem__(self, index):
        image = self.image_data[index] if self.in_memory else self.load_image(index)
        attribute_values = tensor(self.attribute_data[index])
        if self.image_transform:
            attribute_values = self.attribute_transform(attribute_values)
        if self.attribute_transform:
            attribute_values = self.attribute_transform(attribute_values)
        return image, attribute_values

    def split(self, split_name):
        split_partition_indices = {"train": 0, "valid": 1, "test": 2}
        if split_name not in split_partition_indices:
            return
        split_image_file_paths = []
        for image_index, partition_index in enumerate(self.partition_indices):
            if partition_index == split_partition_indices[split_name]:
                split_image_file_paths.append(self.image_file_paths[image_index])
        self.image_file_paths = split_image_file_paths
