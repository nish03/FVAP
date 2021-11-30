from pathlib import Path

from torch import tensor, Generator, randperm
from torchvision.io import read_image

import data.Util
from MultiAttributeDataset import MultiAttributeDataset


class UTKFace(MultiAttributeDataset):
    def __init__(
        self,
        dataset_dir_path,
        image_transform=None,
        attribute_transform=None,
        split_name="all",
        split_seed=42,
        split_train_factor=0.7,
        split_valid_factor=0.2,
    ):
        MultiAttributeDataset.__init__(self, ["age", "gender", "race"], [2, 2, 5])
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(
                f"Invalid dataset directory path {dataset_dir_path} - does not exist"
            )
        self.image_file_paths = sorted(dataset_dir_path.glob("*_*_*_*.jpg"))
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        self.split_name = split_name
        self.image_file_indices = randperm(
            len(self.image_file_paths), generator=Generator().manual_seed(split_seed)
        )
        data_range = range(len(self.image_file_indices))
        train_data_range = range(int(split_train_factor * len(data_range)))
        valid_data_range = range(
            train_data_range.stop,
            train_data_range.stop + int(split_valid_factor * len(data_range)),
        )
        test_data_range = range(valid_data_range.stop, data_range.stop)
        split_data_ranges = {
            "all": data_range,
            "train": train_data_range,
            "valid": valid_data_range,
            "test": test_data_range,
        }
        self.image_file_indices = self.image_file_indices[
            split_data_ranges[self.split_name]
        ]

    def __len__(self):
        return len(self.image_file_indices)

    def __getitem__(self, index):
        image_file_index = self.image_file_indices[index]
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        image_file_name_sections = image_file_path.name.split("_")
        age, gender, race = [int(x) for x in image_file_name_sections[0:3]]
        age = 0 if age < 60 else 1
        attribute_values = tensor([age, gender, race])
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values

    def split(self, **kwargs):
        return data.Util.create_dataset_split(self, **kwargs)
