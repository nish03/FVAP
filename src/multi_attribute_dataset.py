from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from typing import List
from dataclasses import dataclass


@dataclass
class Attribute:
    """
    Attribute is a container that stores meta information, labels and predictions of an attribute during training and
    validation.
    """
    index: int
    name: str
    size: int
    class_counts: List[int] = None
    class_weights: List[float] = None
    targets: torch.Tensor = None
    class_probabilities: torch.Tensor = None
    predictions: torch.Tensor = None


class MultiAttributeDataset(ABC, Dataset):
    """
    MultiAttributeDataset is an abstract baseclass for datasets whose samples consist of images and multiple attributes.

    Deriving classes need to implement the :meth:`_get_sample` method and initialise this base class at construction.
    """
    def __init__(
        self,
        attribute_names: List[str],
        attribute_sizes: List[int],
        attribute_class_counts: List[List[int]],
        prediction_attribute_indices: List[int],
    ):
        """
        Initialises the MultiAttributeDataset

        :param attribute_names: List[str] containing attribute names (this order is used for indexing)
        :param attribute_sizes: List[int] containing the number of classes for each attribute class counts
        :param attribute_class_counts: List[List[int]] containing the frequency of each attributes' class labels in the
            complete dataset
        :param prediction_attribute_indices: List[int] containing the indices of attributes which should be predicted
        """
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_sizes = attribute_sizes
        self.attribute_class_counts = attribute_class_counts
        self.prediction_attribute_indices = prediction_attribute_indices

    def attribute(self, attribute_index: int) -> Attribute:
        """
        Gets the Attribute meta information.

        :param attribute_index: Index of the Attribute
        :return: Attribute meta information (index, name, class count, class frequencies in the complete dataset)
        """
        attribute_name = self.attribute_names[attribute_index]
        attribute_size = self.attribute_sizes[attribute_index]
        attribute_class_counts = self.attribute_class_counts[attribute_index]
        return Attribute(attribute_index, attribute_name, attribute_size, attribute_class_counts)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, int):
        """
        Gets the sample at a given index.

        Uses the abstract method :meth:`_get_sample` from the deriving class to obtain the sample.

        :param index: Index of the sample
        :return: Tensor[attribute_count] containing attribute labels of the sample,
                 Tensor[3, image_height, image_width] containing sample image,
                 Input sample index (for usage with DataLoader)
        """
        attribute_values, image = self._get_sample(index)
        return attribute_values, image, index

    @abstractmethod
    def _get_sample(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        Gets the sample at a given index.

        Abstract method that needs to be implemented by the deriving class.

        :param index: Index of the sample
        :return: Tensor[attribute_count] containing attribute labels of the sample,
                 Tensor[3, image_height, image_width] containing sample image
        """
        raise NotImplementedError()

