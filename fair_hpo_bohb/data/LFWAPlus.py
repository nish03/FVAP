from enum import IntEnum
from pathlib import Path, PureWindowsPath

from torch import tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, CenterCrop
from mat73 import loadmat

import data.Util


class LFWAPlusDataset(Dataset):
    name = "LFWA+"

    class Age(IntEnum):
        Young = 0
        Old = 1

    class Gender(IntEnum):
        Male = 0
        Female = 1

    class WhiteRace(IntEnum):
        NonMember = (0,)
        Member = 1

    class BlackRace(IntEnum):
        NonMember = (0,)
        Member = 1

    class AsianRace(IntEnum):
        NonMember = (0,)
        Member = 1

    class IndianRace(IntEnum):
        NonMember = (0,)
        Member = 1

    target_attributes = [Age, Gender, WhiteRace, BlackRace, AsianRace, IndianRace]

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

        self.lfwa_attributes = loadmat(self.image_dir / "lfw_att_40.mat")
        self.lfwa_plus_attributes = loadmat(self.image_dir / "lfw_att_73.mat")
        self.image_file_paths = self.lfwa_attributes["name"]
        self.transform = Compose(
            [CenterCrop(170)] + ([transform] if transform is not None else [])
        )
        self.target_transform = target_transform
        self.data = []
        self.in_memory = in_memory
        if self.in_memory:
            self.data = [self.get_data(i) for i in range(len(self.image_file_paths))]

    def __len__(self):
        return len(self.image_file_paths)

    def get_data(self, index):
        image_file_path = PureWindowsPath(self.image_file_paths[index])
        image_data = read_image(str(self.image_dir / "lfw" / image_file_path))
        image_data = self.transform(image_data)
        age = int(1 - self.lfwa_attributes["label"][index, 39])
        gender = int(1 - self.lfwa_plus_attributes["label"][index, 0])
        white_race = int(self.lfwa_plus_attributes["label"][index, 2])
        black_race = int(self.lfwa_plus_attributes["label"][index, 3])
        asian_race = int(self.lfwa_plus_attributes["label"][index, 1])
        indian_race = int(self.lfwa_plus_attributes["label"][index, 57])
        target = tensor([age, gender, white_race, black_race, asian_race, indian_race])
        if self.target_transform:
            target = self.target_transform(target)
        return image_data, target

    def __getitem__(self, index):
        if index < len(self.data):
            return self.data[index]
        else:
            return self.get_data(index)

    def split(self, **kwargs):
        return data.Util.create_dataset_split(self, **kwargs)

    @staticmethod
    def load(
        **kwargs,
    ):
        dataset = LFWAPlusDataset(**kwargs)
        return dataset.split()
