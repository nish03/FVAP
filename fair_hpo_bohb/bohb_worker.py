import logging
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from math import log
from os import makedirs, path
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
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import cuda, float32, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder
from hpo.Cost import cost_functions
from model.util.ReconstructionLoss import reconstruction_losses


arg_parser = ArgumentParser(
    description="Perform HPO with BOHB to train a model"
)
if cuda.device_count() > 1:
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
        help="Directory for log files, save states and BOHB output",
    )
else:
    arg_parser.add_argument(
        "-u",
        "--celeba-dir",
        default="C:/Users/OGMENGLI/Projects",
        help="UTKFace dataset directory",
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
    default=144,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
arg_parser.add_argument(
    "--max_epochs",
    default=1,
    type=int,
    required=False,
    help="maximum epochs used for training the model",
)
arg_parser.add_argument(
    "--min_epochs",
    default=1,
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

args = arg_parser.parse_args(argv[1:])
log_file_path = path.join(args.output_dir, "log.txt")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, force=True)
#print(f"Logging started with Output Directory { args.output_dir}")


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
device = "cpu"
#if cudnn.is_available():
#    cudnn.benchmark = True
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)
logging.info(
    f"Data will be loaded with image size {args.image_size} and batch size {args.batch_size}"
)
logging.info(f"Generative models will be trained for maximum {args.max_epochs} epochs and miminum {args.min_epochs} epochs")

logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

image_size = args.image_size
batch_size = args.batch_size
num_workers = device_count * 4
cost_function_name = args.cost
cost_function = cost_functions[cost_function_name]


## the cluster has multiple gpus while the laptop has one GPU only
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

logging.info(f"UTKFace dataset was loaded from directory {dataset_directory}, ")

save_file_directory = path.join(args.output_dir, "save_states")



class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        print("Worker started")
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

    num_iter = 0

    def train_criterion(self, _model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        self.num_iter += 1
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            self.num_iter,
            _data_fraction,
        )





    def validation_criterion(self, _model, _data, _, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        return _model.criterion(
            _data,
            _output,
            _mu,
            _log_var,
            self.num_iter,
            _data_fraction,
        )


    def compute(self, config, budget, working_directory, *args, **kwargs):
        max_iteration = int(budget) * len(train_dataloader)
        config['C_stop_iteration'] = int(config['C_stop_iteration_ratio'] * max_iteration)
        if config['C_stop_iteration'] == 0:
            config['C_stop_iteration'] = 1
        if config['reconstruction_loss'] == 'MS-SSIM':
            config['reconstruction_loss_args']={'window_sigma' : config['ms_ssim_window_sigma']}
        elif config['reconstruction_loss'] == 'LogCosh':
            config['reconstruction_loss_args']={'a' : config['logcosh_a']}
        else:
            config['reconstruction_loss_args'] = {}
        self.runid += 1
        self.budget = budget
        self.current_config = deepcopy(config)
        model = FlexVAE(
            image_size,
            config['latent_dimension_count'],
            config['hidden_layer_count'],
            config['vae_loss_gamma'],
            config['C_max'],
            config['C_stop_iteration'],
            reconstruction_losses[config['reconstruction_loss']],
            config['reconstruction_loss_args']

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
        if isinstance(model, DataParallel):
            _model = model.module
        else:
            _model = model
        train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
            model,
            optimizer,
            lr_scheduler,
            int(budget),
            self.train_criterion,
            self.validation_criterion,
            train_dataloader,
            validation_dataloader,
            #self.save_model_state,
            #self.train_criterion_iteration,
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
                                        'number of parameters': 10 #model.number_of_parameters(),
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

            reconstruction_loss = CSH.CategoricalHyperparameter('reconstruction_loss', ['MAE', 'MSE', 'LogCosh', 'MS-SSIM'])
            ms_ssim_window_sigma = CSH.UniformFloatHyperparameter("ms_ssim_window_sigma", 0.25, 6, default_value=1.5, log=True)
            logcosh_a = CSH.UniformFloatHyperparameter("logcosh_a", 1, 100, default_value=10, log=True)
            cs.add_hyperparameters([reconstruction_loss, ms_ssim_window_sigma, logcosh_a])
            cond1 = CS.EqualsCondition(ms_ssim_window_sigma, reconstruction_loss, 'MS-SSIM')
            cs.add_condition(cond1)
            cond2 = CS.EqualsCondition(logcosh_a, reconstruction_loss, 'LogCosh')
            cs.add_condition(cond2)
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
    #previous_run = hpres.logged_results_to_HBS_result("/srv/nfs/data/mengze/vae/bohb/data")
    bohb = BOHB(configspace=w.get_configspace(),
                run_id='example1', nameserver=host,
                result_logger=result_logger,
                min_budget=args.min_epochs, max_budget=args.max_epochs)
    res = bohb.run(n_iterations=30)
    with open(path.join(args.output_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

