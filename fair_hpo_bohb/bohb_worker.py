
#!/usr/bin/env python

import logging
from argparse import ArgumentParser
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from math import log
from os import makedirs, path
from sys import argv
import torch
from torch.nn.functional import mse_loss
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    Resize,
    ToTensor,
    Lambda,
)
from torchvision.datasets.celeba import CelebA
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.examples.commons import MyWorker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from numpy.random import RandomState
from torch import cuda, float32, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from data.UTKFace import load_utkface
from models.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder



arg_parser = ArgumentParser(
    description="Perform HPO with SMAC to train a generative model"
)

arg_parser.add_argument(
    "-u",
    "--celeba-dir",
    default="/srv/nfs/data/mengze/vae",
    help="UTKFace dataset directory",
    required=False,
),
arg_parser.add_argument(
    "-o",
    "--output-dir",
    default="/srv/nfs/data/mengze/vae/bohb/",
    required=False,
    help="Directory for log files, save states and SMAC output",
)
arg_parser.add_argument(
    "--image_size",
    default=64,
    type=int,
    required=False,
    help="Image size used for loading the dataset",
)
arg_parser.add_argument(
    "--batch_size",
    default=144,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
arg_parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    required=False,
    help="Epochs used for training the generative models",
)
arg_parser.add_argument(
    "--datasplit-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for creating random train, validation and tast dataset splits",
)
arg_parser.add_argument(
    "--smac-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-runtime",
    default=72000,
    type=int,
    required=False,
    help="Runtime used for hyperparameter optimization with SMAC",
)

args = arg_parser.parse_args(argv[1:])

start_date = datetime.now()
#output_directory = path.join(
#    args.output_dir, start_date.strftime("%Y-%m-%d_%H_%M_%S_%f")
#)
makedirs(args.output_dir, exist_ok=True)
log_file_path = path.join(args.output_dir, "log.txt")
# noinspection PyArgumentList
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
print(f"Logging started with Output Directory {args.output_dir}")


logging.info(f"Script started at {start_date}")

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
    logging.info(
        f"Memory allocation was selected to be performed on {device_count} "
        f"CUDA device{'s' if device_count > 1 else ''}"
    )
else:
    device = "cpu"
    device_count = 1
    logging.info("Memory allocation was selected to be performed on the CPU device")

if cudnn.is_available():
    cudnn.benchmark = True
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

image_size = args.image_size
batch_size = args.batch_size
epoch_count = args.epochs
datasplit_seed = args.datasplit_seed
smac_runtime = args.smac_runtime
smac_seed = args.smac_seed
logging.info(
    f"Data will be loaded with image size {image_size} and batch size {batch_size}"
)
logging.info(f"Generative models will be trained for {epoch_count} epochs")
logging.info(
    f"Hyperparameter optimisation with SMAC will be run for {smac_runtime}s and "
    f"with seed {smac_seed}"
)

num_workers = device_count * 4

data_state = {
    "image_size": image_size,
    "batch_size": batch_size,
    "datasplit_seed": datasplit_seed,
}

if args.celeba_dir is not None:
    transform = Compose(
        [
            RandomHorizontalFlip(),
            CenterCrop(148),
            Resize(image_size),
            ToTensor(),
            Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    dataset_directory = args.celeba_dir
    data_state["dataset_directory"] = dataset_directory
    data_state["dataset"] = "celeba"
    train_dataset, validation_dataset, test_dataset = [
        CelebA(root=dataset_directory, split=split, transform=transform, download=False)
        for split in ["train", "valid", "test"]
    ]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(10000))
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(10000))
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(10000))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True#,
        #sampler=train_sampler
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        #sampler=validation_sampler
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        #sampler=test_sampler
    )

    logging.info(f"UTKFace dataset was loaded from directory {dataset_directory}, ")
else:
    raise RuntimeError("No dataset was specified")

save_file_directory = path.join(args.output_dir, "save_states")
#makedirs(save_file_directory)
data_save_file_path = path.join(save_file_directory, "data.pt")
save(data_state, data_save_file_path)


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        print("11111")
        super().__init__(**kwargs)
        # Load the MNIST Data here
        self.train_loader = train_dataloader
        self.validation_loader = validation_dataloader
        self.test_loader = test_dataloader
        self.best_cost = float('inf')
        self.runid = 0
        self.loss = "Reconstruction"
        self.current_config = {}
        self.train_criterion_iteration = 0

    def train_criterion(self, _model, _data, _, _output, _mu, _log_var, _data_fraction):
        self.train_criterion_iteration += 1
        print(self.train_criterion_iteration)
        return FlexVAE.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            self.train_criterion_iteration,
            _data_fraction
        )

    def validation_criterion(self, data, output, mu, log_var, _, data_fraction):
        return FlexVAE.criterion(
            data,
            output,
            mu,
            log_var,
            self.train_criterion_iteration,
            data_fraction
        )

    def compute(self, config, budget, working_directory, *args, **kwargs):
        print(self.runid)
        max_iteration = int(budget) * len(train_dataloader)
        config['C_stop_iteration'] = int(config['C_stop_iteration_ratio'] * max_iteration)
        self.runid += 1
        self.current_config = deepcopy(config)
        model = FlexVAE(
            image_size,
            config['latent_dimension_count'],
            config['hidden_layer_count'],
            config['vae_loss_gamma'],
            config['C_max'],
            config['C_stop_iteration']
        )
        if device_count > 1:
            model = DataParallel(model)
        model.to(device)
        optimizer = Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
        lr_scheduler = ExponentialLR(optimizer, gamma=config['lr_scheduler_gamma'])
        self.train_criterion_iteration += 1
        model, train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
            model,
            optimizer,
            lr_scheduler,
            int(budget),
            model.module.criterion,
            model.module.criterion,
            train_dataloader,
            validation_dataloader,
            self.save_model_state,
            self.train_criterion_iteration,
            schedule_lr_after_epoch=True,
            display_progress=False,
        )
        '''
        for epoch in range(int(budget)):
            loss = 0
            model.train()
            device = next(model.parameters()).device
            for i, x in enumerate(self.train_loader):
                data_fraction = len(x) / len(train_dataloader.dataset)
                output, mu, log_var = model(x)
                loss = mse_loss(x, output)
                loss.backward()
                optimizer.step()
            train_loss = self.evaluate(model, self.train_loader)
            validation_loss = self.evaluate(model, self.validation_loader)
            test_loss = self.evaluate(model, self.test_loader)
        '''


        '''
        final_cost = min(validation_epoch_losses["Reconstruction"])
        '''

        return ({
                'loss': min(validation_epoch_losses['ELBO']), # remember: HpBandSter always minimizes!
                'info': {       'validation_mseloss': min(validation_epoch_losses['Reconstruction']),
                                        'train_mseloss': min(train_epoch_losses['Reconstruction']),
                                        'train_elboloss': min(train_epoch_losses['ELBO']),
                                        'validation_elboloss' : min(validation_epoch_losses['ELBO']),
                                        'train_kldloss' : min(train_epoch_losses['KLD']),
                                        'validation_kldloss' : min(validation_epoch_losses['KLD']),
                                        'number of parameters': 10,
                                }

        })

    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()

            lr = CSH.UniformFloatHyperparameter("lr", 5e-6, 5e-3, default_value=5e-4, log=True)
            weight_decay = CSH.UniformFloatHyperparameter("weight_decay", 0.0, 0.25, default_value=0)
            lr_scheduler_gamma = CSH.UniformFloatHyperparameter("lr_scheduler_gamma", 0.85, 1.0, default_value=0.95)
            cs.add_hyperparameters([lr, weight_decay, lr_scheduler_gamma])

            hidden_layer_count = CSH.UniformIntegerHyperparameter("hidden_layer_count", 1, int(log(image_size, 2)) - 1, default_value=1)
            latent_dimension_count = CSH.UniformIntegerHyperparameter("latent_dimension_count", 16, 512, default_value=128)
            cs.add_hyperparameters([hidden_layer_count, latent_dimension_count])

            vae_loss_gamma = CSH.UniformFloatHyperparameter("vae_loss_gamma",1, 2000, default_value=10, log=True)
            C_max = CSH.UniformFloatHyperparameter("C_max", 5.0, 50.0, default_value=25)
            #max_iteration = budget * len(train_dataloader)
            C_stop_iteration_ratio = CSH.UniformFloatHyperparameter("C_stop_iteration_ratio", 0.05, 1, default_value=0.2)
            cs.add_hyperparameters([vae_loss_gamma, C_max, C_stop_iteration_ratio])
            return cs

    def evaluate(self, model, data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in data_loader:
                output = model(x)
                loss = mse_loss(x,output)
                total_loss += loss
        average_loss = total_loss/len(data_loader.sampler)
        return average_loss

    def save_model_state(self, epoch, model, optimizer, lr_scheduler, train_epoch_losses, validation_epoch_losses):
        cost = validation_epoch_losses["Reconstruction"][-1]
        is_best = self.best_cost > cost
        if is_best:
            self.best_cost = cost

        is_final_epoch = epoch == epoch_count

        if not is_best and not is_final_epoch:
            return

        if isinstance(model, DataParallel):
            model = model.module

        model_state = {
            "runid": self.runid,
            "epoch": epoch,
            "hyperparameter_config": self.current_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "train_epoch_losses": train_epoch_losses,
            "validation_epoch_losses": validation_epoch_losses,
        }

        if is_best:
            model_save_file_name = "model-best.pt"
            model_save_file_path = path.join(save_file_directory, model_save_file_name)
            save(model_state, model_save_file_path)

        if is_final_epoch:
            model_save_file_name = (
                f"model-runid-{self.runid:04}.pt"
            )
            model_save_file_path = path.join(save_file_directory, model_save_file_name)
            save(model_state, model_save_file_path)


# Every process has to lookup the hostname


if __name__ == "__main__":
    worker = PyTorchWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='.')
    print(res)
