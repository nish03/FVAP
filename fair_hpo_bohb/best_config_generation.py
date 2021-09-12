import json
import pickle
from os import path
#import matplotlib.pyplot as plt
from torch import cuda, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize
from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    Resize,
    ToTensor,
    Lambda,
)
from torchvision.datasets.celeba import CelebA
from hpo.Cost import ms_ssim_cost


tweets = []
config_id = []
training_epoch = []
ms_ssim = []
validation_mseloss = []
validation_elboloss = []
validation_kldloss = []
train_mseloss = []
train_elboloss = []
validation_kldloss = []
configs = []
for line in open('configs.json', 'r'):
    configs.append(json.loads(line))
for line in open('results.json', 'r'):
    run_result = json.loads(line)
    tweets.append(run_result)
    config_id.append(run_result[0])
    training_epoch.append(run_result[1])
    ms_ssim.append(run_result[3]['loss'])
    validation_mseloss.append(run_result[3]['info']['validation_mseloss'])
    validation_elboloss.append(run_result[3]['info']['validation_elboloss'])
    validation_kldloss.append(run_result[3]['info']['validation_kldloss'])
    train_mseloss.append(run_result[3]['info']['train_mseloss'])
    train_elboloss.append(run_result[3]['info']['train_elboloss'])
    validation_kldloss.append(run_result[3]['info']['validation_kldloss'])
plot_x = range(len(tweets))
ms_ssim_line = []
min_ssim = 1
for i in plot_x:
    if ms_ssim[i] < min_ssim:
        min_ssim = ms_ssim[i]
    ms_ssim_line.append(min_ssim)
#plt.plot(plot_x, ms_ssim, 'r^', label = 'run_result')
#plt.plot(plot_x, ms_ssim_line, label = 'optimal performance')

min_loss = min(ms_ssim)
bestHype_index = ms_ssim.index(min_loss)
print(bestHype_index)
print(tweets[bestHype_index])
best_config_id = config_id[bestHype_index]
for line in configs:
    if best_config_id == line[0]:
        best_config = line
try:
    print(best_config)
except:
    print("best config not found")


if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
else:
    device = "cpu"
    device_count = 1
device="cpu"
num_workers = 0 #device_count * 4
image_size = 64
batch_size = 144
if device_count > 1:
    transform = Compose(
        [
            RandomHorizontalFlip(),
            CenterCrop(148),
            Resize(image_size),
            ToTensor(),
            Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
else:
    transform = Compose(
        [
            RandomHorizontalFlip(),
            CenterCrop(148),
            Resize(image_size),
            ToTensor(),
            #Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
if device_count > 1:
    dataset_directory = "/srv/nfs/data/mengze/vae"
    output_dir = "/srv/nfs/data/mengze/vae/bohb"
else:
    dataset_directory = "C:/Users/OGMENGLI/Projects"
    output_dir = "C:/Users/OGMENGLI/Projects/HyperFair/fair_hpo_bohb"
try:
    train_dataset, validation_dataset, test_dataset = [
        CelebA(root=dataset_directory, split=split, transform=transform, download=False)
        for split in ["train", "valid", "test"]
    ]
except:
    train_dataset, validation_dataset, test_dataset = [
        CelebA(root="C:/Users/OGMENGLI/Projects", split=split, transform=transform, download=False)
        for split in ["train", "valid", "test"]
    ]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(288))
validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(288))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    sampler=train_sampler
)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    sampler=validation_sampler
)
budget = tweets[bestHype_index][1]
max_iteration = int(budget) * len(train_dataloader)
C_stop_iteration = int(best_config[1]['C_stop_iteration_ratio'] * max_iteration)


model = FlexVAE(
    64,
    128, #best_config[1]['latent_dimension_count'],
    5, #best_config[1]['hidden_layer_count'],
    10, #best_config[1]['vae_loss_gamma'],
    25,#best_config[1]['C_max'],
    10000, #C_stop_iteration
)
#if device_count > 1:
    #model = DataParallel(model)
model.to(device)
optimizer = Adam(
    model.parameters(),
    lr=0.0005, #lr=best_config[1]['lr'],
    weight_decay=0.0, #weight_decay=best_config[1]['weight_decay'],
)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)#best_config[1]['lr_scheduler_gamma'])

if isinstance(model, DataParallel):
    _model = model.module
else:
    _model = model
budget = 20

if __name__ == '__main__':
    def train_criterion(_model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        train_criterion.iteration += 1
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )


    train_criterion.iteration = 0


    def validation_criterion(_model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )


    train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
        model,
        optimizer,
        lr_scheduler,
        int(budget),
        train_criterion,
        validation_criterion,
        train_dataloader,
        validation_dataloader,
        schedule_lr_after_epoch=True,
        display_progress=False,
    )
    cost, additional_info = ms_ssim_cost(_model, validation_dataloader, model)
    model_state = {
        "hyperparameters": best_config[1],
        "cost": cost,
        "additional_info": additional_info,
        "epoch_count": int(budget),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_save_file_name = f"model-run-{bestHype_index:04}.pt"
    model_save_file_path = path.join(output_dir, model_save_file_name)
    save(model_state, model_save_file_path)
