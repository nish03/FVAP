from csv import reader
from pathlib import Path

import numpy
import torch

from multi_attribute_dataset import MultiAttributeDataset
from numpy import arange, loadtxt, unique
from torch import tensor
from torchvision.io import read_image


class CelebA(MultiAttributeDataset):
    """
    CelebA provides access to the CelebFaces Attributes Dataset as an :class:`MultiAttributeDataset`.
    
    See https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html for more information on this dataset.
    This class requires the aligned and cropped images (img_align_celeba), the attribute (list_attr_celeba.txt) and the
    eval partition (list_eval_partition.txt) files inside a common dataset directory.
    """
    def __init__(
        self,
        dataset_dir_path: Path,
        image_transform=None,
        attribute_transform=None,
        split_name: str = "all",
    ):
        """
        Creates a new CelebA dataset instance.

        :param dataset_dir_path: Path pointing to the location of the dataset directory
        :param image_transform: Transformation(Tensor[3, 32, 32]) that is applied to each image
        :param attribute_transform: Transformation(Tensor[40]) that is applied to the attributes of each image
        :param split_name: Name of the dataset partition ("all", "train", "valid" or "test")
        """
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
        """
        Gets the amount of samples inside the selected partition.

        :return: int Dataset partition sample count
        """
        return self.attribute_data.shape[0]

    def _get_sample(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        Gets the sample data for a given index.

        :param index: Sample index
        :return: Tensor[3, 218, 178] containing the sample image,
                 ndarray[40] containing the sample attribute class labels
            Types can vary if image_transform and attribute_transform were set during construction
        """
        image_file_path = self.image_dir_path / f"{self.image_file_numbers[index]:0>6}.jpg"
        image = read_image(str(image_file_path))
        attribute_values = tensor(self.attribute_data[index])
        if self.image_transform:
            image = self.image_transform(image)
        if self.attribute_transform:
            attribute_values = self.attribute_transform(attribute_values)
        return image, attribute_values

    def _split_image_indices(self, partition_indices: numpy.ndarray, split_name: str):
        """
        Gets the image indices for a dataset split.

        :param partition_indices: numpy.ndarray[dataset_sample_count]
        :param split_name: Name of the dataset split ("train", "valid" or "test")
        :return: numpy.ndarray[datset_split_sample_count] containing sample indices from the selected dataset split
        """
        split_name_to_partition_index = {"train": 0, "valid": 1, "test": 2}
        split_image_indices = arange(self.dataset_image_count)
        if split_name in split_name_to_partition_index:
            split_image_indices = split_image_indices[partition_indices == split_name_to_partition_index[split_name]]
        return split_image_indices
