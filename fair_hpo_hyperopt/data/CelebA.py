from os import path
from enum import IntEnum

from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, PILToTensor, CenterCrop, Lambda
from torchvision.datasets.utils import verify_str_arg


class CelebADataset(Dataset):
    class Age(IntEnum):
        Young = 0
        Old = 1

    class Gender(IntEnum):
        Male = 0
        Female = 1

    target_attributes = [Age, Gender]

    def __init__(
        self,
        image_directory_path, # pass the folder that includes /celeba/img_align_celeba.zip, for ex: /home/erdem/dataset
        split="all",
        transform=None,
        target_transform=None,
        in_memory=False,
    ):
        self.image_directory = image_directory_path
        if not path.isdir(self.image_directory):
            raise ValueError(
                f"Invalid image directory path {image_directory_path} - does not exist"
            )
        get_sensitve_attributes = Lambda(lambda x: 1 - x[[39, 20]])
        crop_images = Compose([CenterCrop(148), PILToTensor()])
        self.celeba = CelebA(
            root=self.image_directory,
            split=split,
            transform=crop_images,
            target_transform=get_sensitve_attributes,
            download=False,
        )

        # workaround for splits not being applied to images
        # fixed future torchvision version
        # see https://github.com/pytorch/vision/pull/4377
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_index = split_map[
            verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))
        ]
        splits = self.celeba._load_csv("list_eval_partition.txt")
        self.celeba.filename = (
            splits.index
            if split_index is None
            else [
                splits.index[i]
                for i in (splits.data == split_index).squeeze().nonzero().squeeze()
            ]
        )

        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        if in_memory:
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


def load_celeba(**kwargs):
    train_dataset = CelebADataset(split="train", **kwargs)
    validation_dataset = CelebADataset(split="valid", **kwargs)
    test_dataset = CelebADataset(split="test", **kwargs)
    return train_dataset, validation_dataset, test_dataset
