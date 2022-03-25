from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from typing import List
from dataclasses import dataclass


class MultiAttributeDataset(ABC, Dataset):
    def __init__(
        self,
        attribute_names: List[str],
        attribute_sizes: List[int],
        prediction_attribute_indices: List[int],
    ):
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_sizes = attribute_sizes
        self.prediction_attribute_indices = prediction_attribute_indices

    def attribute(self, attribute_index):
        attribute_name = self.attribute_names[attribute_index]
        attribute_size = self.attribute_sizes[attribute_index]
        return Attribute(attribute_index, attribute_name, attribute_size)

    @abstractmethod
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError()


@dataclass
class Attribute:
    index: int
    name: str
    size: int
    targets: torch.Tensor = None
    class_probabilities: torch.Tensor = None
    predictions: torch.Tensor = None
