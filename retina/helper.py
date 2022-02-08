import pandas as pd
from torch.utils.data import Dataset
from torch import tensor, round, where
from PIL import Image
import os


class retinaDataset(Dataset):
    def __init__(self, transform,
                 imagepath="../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped",
                 total=None):
        self.df = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels_cropped.csv")

        if (total is not None):
            self.df = self.df[:total]

        self.transform = transform

        self.imagepath = imagepath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.imagepath, self.df.iloc[index].image + ".jpeg")
        img = Image.open(img_path)

        if (self.transform):
            img = self.transform(img)

        return img, tensor(self.df.iloc[index].level)


def continuous_to_categorical(predictions):
    predictions = round(predictions)
    for id in range(len(predictions)):
        if predictions[id] < 0:
            predictions[id] = float(0)
        if predictions[id] > 4:
            predictions[id] = float(4)

    return predictions.squeeze()