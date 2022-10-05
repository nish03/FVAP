from pathlib import Path

import torch
from torch import Generator, randperm, tensor
from torchvision.io import read_image
from pandas import DataFrame

from multi_attribute_dataset import MultiAttributeDataset


class UTKFace(MultiAttributeDataset):
    """
    UTKFace Dataset as a :class:`MultiAttributeDataset`

    See https://susanqq.github.io/UTKFace/ for more information on this dataset.
    This class requires the aligned and cropped images with the original file names
    (e.g. "116_13_20170120134744096.jpg.chip.jpg") for attribute parsing.

    We derive three categorical attributes from the original labels:
        age_group (derived from the original age values)
            0 - 30 or fewer years old
            1 - between 31 and 60 years old
            2 - 61 or more years old
        gender
            0 - Male
            1 - Female
        ethnicity:
            0 - White
            1 - Black
            2 - Asian
            3 - Indian
            4 - Others (like Hispanic, Latino, Middle Eastern)
    """
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
        """
        Creates a new UTKFace dataset instance.

        :param dataset_dir_path: Path pointing to the location of the dataset directory
        :param image_transform: Transformation(Tensor[3, 32, 32]) that is applied to each image
        :param attribute_transform: Transformation(Tensor[40]) that is applied to the attributes of each image
        :param split_name: Name of the dataset partition ("all", "train", "valid" or "test")
        :param split_seed: Seed for the random dataset split generation
        :param split_train_factor: Fraction of samples in the train partition
        :param split_valid_factor: Fraction of samples in the validation partition
        """
        self.dataset_dir_path = Path(dataset_dir_path)
        if not self.dataset_dir_path.is_dir():
            raise ValueError(f"Invalid dataset directory path {dataset_dir_path} - does not exist")
        self.image_file_paths = sorted(dataset_dir_path.glob("*_*_*_*.jpg"))
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        self.split_name = split_name
        self.image_file_indices = randperm(len(self.image_file_paths), generator=Generator().manual_seed(split_seed))
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
        self.image_file_indices = self.image_file_indices[split_data_ranges[self.split_name]]
        attribute_names = ["age_group", "gender", "ethnicity"]
        self.attribute_data = {attribute_name: [] for attribute_name in attribute_names}
        for image_file_index in self.image_file_indices:
            image_file_path = self.image_file_paths[image_file_index]
            image_file_name_sections = image_file_path.name.split("_")
            age, gender, ethnicity = [int(x) for x in image_file_name_sections[0:3]]
            if age <= 30:
                age_group = 0
            elif 31 <= age <= 60:
                age_group = 1
            else:
                age_group = 2
            self.attribute_data["age_group"].append(age_group)
            self.attribute_data["gender"].append(gender)
            self.attribute_data["ethnicity"].append(ethnicity)
        self.attribute_data = DataFrame(self.attribute_data)
        attribute_sizes = [3, 2, 5]
        attribute_class_counts = [
            self.attribute_data[attribute_name].value_counts().tolist() for attribute_name in attribute_names
        ]
        prediction_attribute_indices = [0, 1, 2]
        MultiAttributeDataset.__init__(
            self, attribute_names, attribute_sizes, attribute_class_counts, prediction_attribute_indices
        )

    def __len__(self) -> int:
        """
        Gets the amount of samples inside the selected partition.

        :return: int Dataset partition sample count
        """
        return len(self.image_file_indices)

    def _get_sample(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        Gets the sample data for a given index.

        :param index: Sample index
        :return: Tensor[3, 200, 200] containing the sample image,
                 ndarray[3] containing the sample attribute class labels (age_group, gender, ethnicity)
            Types can vary if image_transform and attribute_transform were set during construction
        """
        image_file_index = self.image_file_indices[index]
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        attribute_values = tensor(self.attribute_data.iloc[index].tolist())
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values
