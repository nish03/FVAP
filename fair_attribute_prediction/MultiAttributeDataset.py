from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from typing import List


class MultiAttributeDataset(ABC, Dataset):
    def __init__(self, attribute_names: List[str], attribute_sizes: List[int]):
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_sizes = attribute_sizes

    @abstractmethod
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError()
