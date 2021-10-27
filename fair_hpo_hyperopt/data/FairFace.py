from enum import IntEnum
from itertools import chain, islice
from collections import defaultdict
from csv import reader
from pathlib import Path

from torch import tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from data.util.DatasetSplit import create_dataset_split


class FairFaceDataset(Dataset):
    name = "FairFace"

    class Age(IntEnum):
        Young = 0
        Old = 1

    class Gender(IntEnum):
        Male = 0
        Female = 1

    class Race(IntEnum):
        White = 0
        MiddleEastern = 1
        EastAsian = 2
        SoutheastAsian = 3
        Black = 4
        Indian = 5
        LatinoHispanic = 6

    target_attributes = [Age, Gender, Race]

    def __init__(
        self,
        image_dir_path,
        transform=None,
        target_transform=None,
        in_memory=False,
    ):
        self.image_dir = Path(image_dir_path)
        if not self.image_dir.is_dir():
            raise ValueError(
                f"Invalid image directory path {self.image_dir} - does not exist"
            )

        labels = defaultdict(list)
        with open(self.image_dir / "train_labels.csv") as train_labels_file, open(
            self.image_dir / "val_labels.csv"
        ) as val_labels_file:
            train_labels_reader = reader(train_labels_file)
            val_labels_reader = reader(val_labels_file)
            key_row = next(train_labels_reader)
            for label_row in chain(
                train_labels_reader, islice(val_labels_reader, 1, None)
            ):
                for key_index, key in enumerate(key_row):
                    labels[key].append(label_row[key_index])
        age_label_map = {
            key: self.Age.Young
            for key in ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59"]
        }
        age_label_map.update({key: self.Age.Old for key in ["60-69", "more than 70"]})
        gender_label_map = {"Female": self.Gender.Female, "Male": self.Gender.Male}
        race_label_map = {
            "White": self.Race.White,
            "Middle Eastern": self.Race.MiddleEastern,
            "East Asian": self.Race.EastAsian,
            "Southeast Asian": self.Race.SoutheastAsian,
            "Black": self.Race.Black,
            "Indian": self.Race.Indian,
            "Latino_Hispanic": self.Race.LatinoHispanic,
        }
        self.fairface_attributes = {
            "age": list(map(lambda label: int(age_label_map[label]), labels["age"])),
            "gender": list(
                map(lambda label: int(gender_label_map[label]), labels["gender"])
            ),
            "race": list(map(lambda label: int(race_label_map[label]), labels["race"])),
        }
        self.image_file_paths = labels["file"]
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.in_memory = in_memory
        if self.in_memory:
            self.data = [self.get_data(i) for i in range(len(self.image_file_paths))]

    def __len__(self):
        return len(self.image_file_paths)

    def get_data(self, index):
        image_file_path = Path(self.image_file_paths[index])
        image_data = read_image(str(self.image_dir / image_file_path))
        if self.transform:
            image_data = self.transform(image_data)
        age = self.fairface_attributes["age"][index]
        gender = self.fairface_attributes["gender"][index]
        race = self.fairface_attributes["race"][index]
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

    @staticmethod
    def load(
        **kwargs,
    ):
        dataset = FairFaceDataset(**kwargs)
        return dataset.split()
