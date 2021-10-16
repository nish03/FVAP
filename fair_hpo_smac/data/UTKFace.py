from glob import glob
from os import path
from enum import IntEnum

from torch import tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from data.util.DatasetSplit import create_dataset_split


class UTKFaceDataset(Dataset):
    name = "UTKFace"

    class Age(IntEnum):
        Young = 0
        Old = 1

    class Gender(IntEnum):
        Male = 0
        Female = 1

    class Race(IntEnum):
        White = 0
        Black = 1
        Asian = 2
        Indian = 3
        Other = 4

    target_attributes = [Age, Gender, Race]

    def __init__(
        self,
        image_dir_path,
        transform=None,
        target_transform=None,
        in_memory=False,
    ):
        self.image_dir = image_dir_path
        if not path.isdir(self.image_dir):
            raise ValueError(
                f"Invalid image directory path {image_dir_path} - does not exist"
            )
        self.image_file_paths = glob(path.join(image_dir_path, "*_*_*_*.jpg"))
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.in_memory = in_memory
        if self.in_memory:
            self.data = [self.get_data(i) for i in range(len(self.image_file_paths))]

    def __len__(self):
        return len(self.image_file_paths)

    def get_data(self, index):
        image_file_path = self.image_file_paths[index]
        image_data = read_image(image_file_path)
        if self.transform:
            image_data = self.transform(image_data)
        image_file_name = path.basename(image_file_path)
        image_file_name_sections = image_file_name.split("_")
        age, gender, race = [int(x) for x in image_file_name_sections[0:3]]
        age = 0 if age < 60 else 1
        target = tensor([age, gender, race])
        if self.target_transform:
            target = self.target_transform(target)
        return image_data, target

    def __getitem__(self, index):
        if index < len(self.data):
            return self.data[index]
        else:
            return self.get_data(index)

    def split(self, **kwargs):
        return create_dataset_split(self, **kwargs)


def load_utkface(
    train_split_factor=0.7, validation_split_factor=0.2, random_split_seed=42, **kwargs
):
    dataset = UTKFaceDataset(**kwargs)
    return dataset.split(
        train_split_factor=train_split_factor,
        validation_split_factor=validation_split_factor,
        random_split_seed=random_split_seed,
    )
