import torch
from torch import cuda
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, random_split
from torch.backends import cudnn
from torchvision import transforms
from torchvision.models import inception_v3, resnet50
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from helper import retinaDataset, continuous_to_categorical

batch_size = 32
learning_rate = 1e-4

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()

else:
    device = "cpu "
    device_count = 1
num_workers = device_count * 4
if cudnn.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False

cropped = pd.read_csv("trainLabels_cropped.csv")
images_path = "resized_train_cropped"

my_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = retinaDataset(transform=my_transform)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
valid_size = dataset_size - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

model = inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.fc = torch.nn.Linear(in_features=2048, out_features=1, bias=True)
model.aux_logits = False
model = model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_criterion = MSELoss()

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []
for epoch in range(10):
    print("epoch", epoch)
    model.train()
    for data, target in tqdm(train_dataloader):
        data = data.to(device=device)
        target = target.to(device=device)
        score = model(data)
        optimizer.zero_grad()
        target = target.float()
        loss = loss_criterion(score.squeeze(), target)
        loss.backward()
        optimizer.step()

    model.eval()

    train_correct_output = 0
    train_total_output = 0
    train_loss_epoch = []
    with torch.no_grad():
        for x, y in tqdm(train_dataloader):
            x = x.to(device=device)
            y = y.to(device=device)

            score = model(x)
            _, predictions = score.max(1)
            predictions = continuous_to_categorical(score)
            train_correct_output += (y == predictions).sum()
            train_total_output += predictions.shape[0]
            target = y.float()
            loss = loss_criterion(score.squeeze(), target)
            train_loss_epoch.append(loss.item())
        train_acc = float(train_correct_output / train_total_output) * 100
        train_loss_epoch = np.mean(train_loss_epoch)
        print(f"training accuracy: {train_acc:.2f}")
        print(f"training loss: {train_loss_epoch:.2f}")
        train_accuracy.append(train_acc)
        train_loss.append(train_loss_epoch)

    valid_correct_output = 0
    valid_total_output = 0
    valid_loss_epoch = []
    with torch.no_grad():
        for x, y in tqdm(valid_dataloader):
            x = x.to(device=device)
            y = y.to(device=device)

            score = model(x)
            _, predictions = score.max(1)
            predictions = continuous_to_categorical(score)
            valid_correct_output += (y == predictions).sum()
            valid_total_output += predictions.shape[0]

            target = y.float()
            loss = loss_criterion(score.squeeze(), target)
            valid_loss_epoch.append(loss.item())
        valid_acc = float(valid_correct_output / valid_total_output) * 100
        valid_loss_epoch = np.mean(valid_loss_epoch)
        print(f"validation accuracy: {valid_acc:.2f}")
        print(f"validation loss: {valid_loss_epoch:.2f}")
        valid_accuracy.append(valid_acc)
        valid_loss.append(valid_loss_epoch)
print(train_accuracy, valid_accuracy)
print(train_loss, valid_loss)