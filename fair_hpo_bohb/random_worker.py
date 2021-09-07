import logging
from argparse import ArgumentParser
from math import log
from os import path
import pickle
from sys import argv
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
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB, RandomSearch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import cuda
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder
from hpo.Cost import cost_functions



arg_parser = ArgumentParser(
    description="Perform HPO with BOHB to train a model"
)
# cluster has more than one GPU while the laptop has one GPU
if cuda.device_count() > 1:
    arg_parser.add_argument(
        "-u",
        "--celeba-dir",
        default="/srv/nfs/data/mengze/vae",
        help="dataset directory",
        required=False,
    ),
    arg_parser.add_argument(
        "-o",
        "--output-dir",
        default="/srv/nfs/data/mengze/vae/bohb/",
        required=False,
        help="Directory for log files, save states and BOHB output",
    )
else:
    arg_parser.add_argument(
        "-u",
        "--celeba-dir",
        default="C:/Users/OGMENGLI/Projects",
        help="dataset directory",
        required=False,
    ),
    arg_parser.add_argument(
        "-o",
        "--output-dir",
        default="C:/Users/OGMENGLI/Projects/HyperFair/fair_hpo_bohb",
        required=False,
        help="Directory for log files, save states and BOHB output",
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
    default=256,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
arg_parser.add_argument(
    "--max_epochs",
    default=15,
    type=int,
    required=False,
    help="maximum epochs used for training the model",
)
arg_parser.add_argument(
    "--min_epochs",
    default=5,
    type=int,
    required=False,
    help="minimum epochs used for training the model",
)
arg_parser.add_argument(
    '--worker',
    help='Flag to turn this into a worker process',
    action='store_true'
)
arg_parser.add_argument(
    "--cost",
    default="MS-SSIM",
    choices=["MS-SSIM", "FairMS-SSIM"],
    required=False,
    help="Cost function used for HPO",
)
args = arg_parser.parse_args(argv[1:])
log_file_path = path.join(args.output_dir, "log.txt")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, force=True)
logging.info(f"Script started")
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
logging.info(
    f"Data will be loaded with image size {args.image_size} and batch size {args.batch_size}"
)
logging.info(f"Generative models will be trained for maximum {args.max_epochs} epochs and miminum {args.min_epochs} epochs")

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
else:
    device = "cpu"
    device_count = 1

if cudnn.is_available():
    cudnn.benchmark = True
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

image_size = args.image_size
batch_size = args.batch_size
num_workers = device_count * 4
cost_function_name = args.cost
cost_function = cost_functions[cost_function_name]


## windows environment does not support doing lambda in training
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
dataset_directory = args.celeba_dir
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
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(100))
validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(100))
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(100))
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
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    #sampler=test_sampler
)

logging.info(f"Dataset was loaded from directory {dataset_directory}, ")

save_file_directory = path.join(args.output_dir, "save_states")



class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        print("Worker started")
        super().__init__(**kwargs)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        max_iteration = int(budget) * len(train_dataloader)
        config['C_stop_iteration'] = int(config['C_stop_iteration_ratio'] * max_iteration)
        self.budget = budget
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
        if isinstance(model, DataParallel):
            _model = model.module
        else:
            _model = model
        model, train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
            model,
            optimizer,
            lr_scheduler,
            int(budget),
            _model.criterion,
            _model.criterion,
            train_dataloader,
            validation_dataloader,
            schedule_lr_after_epoch=True,
            display_progress=False,
        )

        cost, additional_info = cost_function(_model, validation_dataloader, model)

        return ({
                'loss': cost,
                #'loss': min(validation_epoch_losses['ELBO']), # remember: HpBandSter always minimizes!
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

if __name__ == "__main__":
    host = '127.0.0.1'
    if args.worker:
        w = PyTorchWorker(run_id="example1", nameserver=host)
        # w.load_nameserver_credentials(working_directory=output_directory)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=args.output_dir, overwrite=True)
    NS = hpns.NameServer(run_id='example1', host=host, port=None)
    NS.start()
    w = PyTorchWorker(nameserver=host, run_id='example1')
    w.run(background=True)
    rs = RandomSearch(configspace=w.get_configspace(),
                run_id='example1', nameserver=host,
                result_logger=result_logger,
                min_budget=args.min_epochs, max_budget=args.max_epochs
                )
    res = rs.run(n_iterations=1)
    with open(path.join(args.output_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    rs.shutdown(shutdown_workers=True)
    NS.shutdown()

