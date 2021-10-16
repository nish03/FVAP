from torch import Generator
from torch.utils.data import random_split


def create_dataset_split(
    dataset, train_split_factor=0.7, validation_split_factor=0.2, random_split_seed=42
):
    train_count = int(train_split_factor * len(dataset))
    validation_count = int(validation_split_factor * len(dataset))
    test_count = len(dataset) - train_count - validation_count
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [train_count, validation_count, test_count],
        Generator().manual_seed(random_split_seed),
    )
    return train_dataset, validation_dataset, test_dataset
