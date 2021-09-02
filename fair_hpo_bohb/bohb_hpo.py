from bohb_worker import PyTorchWorker as worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import logging
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from math import log
from os import makedirs, path
from sys import argv
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
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
import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging

arg_parser = ArgumentParser(
    description="Perform HPO with SMAC to train a generative model"
)


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
    default=128,
    type=int,
    required=False,
    help="Epochs used for training the generative models",
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
arg_parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

args = arg_parser.parse_args()
start_date = datetime.now()
output_directory = path.join(
    args.output_dir, start_date.strftime("%Y-%m-%d_%H_%M_%S_%f")
)
makedirs(output_directory, exist_ok=True)
log_file_path = path.join(output_directory, "log.txt")
# noinspection PyArgumentList
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
print(f"Logging started with Output Directory {output_directory}")

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

image_size = args.image_size
batch_size = args.batch_size
epoch_count = args.epochs
smac_runtime = args.smac_runtime
smac_seed = args.smac_seed
logging.info(
    f"Data will be loaded with image size {image_size} and batch size {batch_size}"
)
logging.info(f"Generative models will be trained for maximum {epoch_count} epochs")

num_workers = device_count * 4

if __name__ == "__main__":
    host = '127.0.0.1'
    
    if args.worker:
        w = worker(run_id="example1", nameserver=host)
        #w.load_nameserver_credentials(working_directory=output_directory)
        w.run(background=False)
        exit(0)
    
    result_logger = hpres.json_result_logger(directory=args.output_dir, overwrite=True)
    '''
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1',  port=0, working_directory=args.output_dir)
    ns_host, ns_port = NS.start()
    w = worker(run_id='example1', host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    w.run(background=True)
    bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = 'example1', host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port,
                  result_logger=result_logger, min_budget=3, max_budget=5
               )
    res = bohb.run(n_iterations=10)
    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    '''
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    #w = worker( nameserver='127.0.0.1', run_id='example1')
    #w.run(background=True)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(configspace=worker.get_configspace(),
                run_id='example1',
                min_budget=6, max_budget=54
                )
    res = bohb.run(n_iterations=100, min_n_workers=2)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    
