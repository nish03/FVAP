from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class MultiAttributeDataset(ABC, Dataset):
    def __init__(self, attribute_names, attribute_class_counts):
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_class_counts = attribute_class_counts
        self.attribute_class_score_ranges = [
            range(
                range_start := sum(attribute_class_counts[0:attribute_index]),
                range_start + attribute_class_counts[attribute_index],
            )
            for attribute_index in range(self.attribute_count)
        ]

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
