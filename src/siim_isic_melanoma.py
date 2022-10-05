from pathlib import Path

import torch
from torch import Generator, randperm, tensor
from torchvision.io import read_image
from pandas import read_csv, DataFrame

from multi_attribute_dataset import MultiAttributeDataset


class SIIMISICMelanoma(MultiAttributeDataset):
    """
    SIIMISICMelanoma provides access to the dataset from the SIIM-ISIC-Melanoma classification challenge as an
    :class:`MultiAttributeDataset`.
    
    See https://www.kaggle.com/competitions/siim-isic-melanoma-classification for more information on this dataset.
    This class requires the JPEG images (such as jpeg/train/ISIC_0015719.jpg) and attribute labels (train.csv) for
    training. The test sample data isn't required it doesn't contain age and gender annotations. This class can create
    new dateset splits from the original training samples instead.

    We derive three categorical attributes from the original labels:
        age_group (derived from the original age_approx values)
            0 - 30 or fewer years old
            1 - between 31 and 60 years old
            2 - 61 or more years old
        gender
            0 - Male
            1 - Female
        diagnosis:
            0 - Benign (noncancerous)
            1 - Malignant (cancerous)
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
        Creates a new SIIMISICMelanoma dataset instance.

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
        image_directory_path = dataset_dir_path / "jpeg" / "train"
        data_file_path = dataset_dir_path / "train.csv"
        self.image_file_paths = sorted(image_directory_path.glob("*.jpg"))
        self.image_transform = image_transform
        self.attribute_transform = attribute_transform
        self.split_name = split_name
        self.image_file_indices = randperm(len(self.image_file_paths), generator=Generator().manual_seed(split_seed))
        image_range = range(len(self.image_file_indices))
        train_image_range = range(int(split_train_factor * len(image_range)))
        valid_image_range = range(
            train_image_range.stop,
            train_image_range.stop + int(split_valid_factor * len(image_range)),
        )
        test_image_range = range(valid_image_range.stop, image_range.stop)
        split_image_ranges = {
            "all": image_range,
            "train": train_image_range,
            "valid": valid_image_range,
            "test": test_image_range,
        }
        self.image_file_indices = self.image_file_indices[split_image_ranges[self.split_name]].tolist()
        raw_attribute_data = read_csv(data_file_path).sort_values(by="image_name")
        raw_attribute_data["sex"].fillna(raw_attribute_data["sex"].mode()[0], inplace=True)
        raw_attribute_data["age_approx"].fillna(raw_attribute_data["age_approx"].mode()[0], inplace=True)
        attribute_names = ["age_group", "gender", "diagnosis"]
        self.attribute_data = {attribute_name: [] for attribute_name in attribute_names}
        for image_file_index in self.image_file_indices:
            sex, age_approx, target = raw_attribute_data.iloc[image_file_index, [2, 3, 7]]
            if age_approx <= 30:
                age = 0
            elif 31 <= age_approx <= 60:
                age = 1
            else:
                age = 2
            if sex == "male":
                gender = 0
            else:
                gender = 1
            diagnosis = int(target)
            self.attribute_data[attribute_names[0]].append(age)
            self.attribute_data[attribute_names[1]].append(gender)
            self.attribute_data[attribute_names[2]].append(diagnosis)
        self.attribute_data = DataFrame(self.attribute_data)
        attribute_sizes = [3, 2, 2]
        attribute_class_counts = [
            self.attribute_data[attribute_name].value_counts().tolist() for attribute_name in attribute_names
        ]
        prediction_attribute_indices = [2]
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
        :return: Tensor[3, image_height, image_width] containing the sample image,
                 ndarray[3] containing the sample attribute class labels (age_group, gender, ethnicity)
            Types can vary if image_transform and attribute_transform were set during construction
        """
        image_file_index = self.image_file_indices[index]
        image_file_path = self.image_file_paths[image_file_index]
        image = read_image(str(image_file_path))
        if self.image_transform:
            image = self.image_transform(image)
        age, gender, diagnosis = self.attribute_data.iloc[index]
        attribute_values = tensor([age, gender, diagnosis])
        if self.attribute_transform:
            attribute_values = self.target_transform(attribute_values)
        return image, attribute_values
