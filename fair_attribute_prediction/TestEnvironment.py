#!/usr/bin/env python
from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype

from SlimCNN import SlimCNN
from Training import loss_with_metrics
from UTKFace import UTKFace
from Util import get_device, get_device_count


num_workers = min(4 * get_device_count(), 8)
batch_size = 32
learning_rate = 0.1
utkface_dataset_dir = Path("datasets") / "UTKFace"

train_transform = ConvertImageDtype(torch.float32)
train_dataset = UTKFace(
    dataset_dir_path=utkface_dataset_dir,
    image_transform=train_transform,
    split_name="train",
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

model = SlimCNN(attribute_sizes=train_dataset.attribute_sizes)
model = DataParallel(model)
model.to(get_device())

optimizer = Adam(model.parameters(), lr=learning_rate)

parameters = {
    "sensitive_attribute": train_dataset.attribute(0),
    "target_attribute": train_dataset.attribute(1),
    "fair_loss_weight": 1.0,
    "fair_loss_type": "iou",
}

model.train()

train_metrics_state = None
for train_batch_data in train_dataloader:
    optimizer.zero_grad(set_to_none=True)

    loss, train_metrics, train_metrics_state = loss_with_metrics(
        model, train_batch_data, train_metrics_state, parameters
    )

    if loss.isnan():
        raise RuntimeError("Encountered NaN loss")

    loss.backward()
    optimizer.step()

    print(f"{train_metrics}")

