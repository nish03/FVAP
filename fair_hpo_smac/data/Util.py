from pathlib import Path

from data.CelebA import CelebADataset
from data.FairFace import FairFaceDataset
from data.LFWAPlus import LFWAPlusDataset
from data.UTKFace import UTKFaceDataset


from torch import Generator
from torch.utils.data import random_split

datasets = {
    dataset.name: dataset
    for dataset in [CelebADataset, FairFaceDataset, LFWAPlusDataset, UTKFaceDataset]
}


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


def load_dataset(dataset_name, dataset_dir=None, **kwargs):
    dataset_class = datasets[dataset_name]

    if dataset_dir is None:
        dataset_dir = Path("datasets") / dataset_name

    return (
        dataset_class.load(image_dir_path=dataset_dir, **kwargs),
        dataset_class,
        dataset_dir,
    )
