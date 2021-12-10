from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class MultiAttributeDataset(ABC, Dataset):
    def __init__(self, attribute_names: list[str], attribute_sizes: list[int]):
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_sizes = attribute_sizes

    @abstractmethod
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError()
