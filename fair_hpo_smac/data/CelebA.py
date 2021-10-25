from os import path
from enum import IntEnum

from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, PILToTensor, CenterCrop, Lambda


class CelebADataset(Dataset):
    name = "CelebA"

    class Age(IntEnum):
        Young = 0
        Old = 1

    class Gender(IntEnum):
        Male = 0
        Female = 1

    target_attributes = [Age, Gender]

    def __init__(
        self,
        image_dir_path,
        split="all",
        transform=None,
        target_transform=None,
        in_memory=False,
    ):
        self.image_dir = image_dir_path
        if not path.isdir(self.image_dir):
            raise ValueError(
                f"Invalid image directory path {image_dir_path} - does not exist"
            )
        get_sensitve_attributes = Lambda(lambda x: 1 - x[[39, 20]])
        crop_images = Compose([CenterCrop(148), PILToTensor()])

        self.celeba = CelebA(
            root=self.image_dir,
            split=split,
            transform=crop_images,
            target_transform=get_sensitve_attributes,
            download=False,
        )

        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.in_memory = in_memory
        if self.in_memory:
            self.data = [self.get_data(i) for i in range(len(self.celeba))]

    def __len__(self):
        return len(self.celeba)

    def get_data(self, index):
        image_data, target = self.celeba[index]
        if self.transform:
            image_data = self.transform(image_data)
        if self.target_transform:
            target = self.target_transform(target)
        return image_data, target

    def __getitem__(self, index):
        if index < len(self.data):
            return self.data[index]
        else:
            return self.get_data(index)

    def split(self):
        train_dataset = CelebADataset(
            image_dir_path=self.image_dir,
            split="train",
            transform=self.transform,
            target_transform=self.target_transform,
            in_memory=self.in_memory,
        )
        validation_dataset = CelebADataset(
            image_dir_path=self.image_dir,
            split="valid",
            transform=self.transform,
            target_transform=self.target_transform,
            in_memory=self.in_memory,
        )
        test_dataset = CelebADataset(
            image_dir_path=self.image_dir,
            split="test",
            transform=self.transform,
            target_transform=self.target_transform,
            in_memory=self.in_memory,
        )
        return train_dataset, validation_dataset, test_dataset


def load_celeba(**kwargs):
    train_dataset = CelebADataset(split="train", **kwargs)
    validation_dataset = CelebADataset(split="valid", **kwargs)
    test_dataset = CelebADataset(split="test", **kwargs)
    return train_dataset, validation_dataset, test_dataset
