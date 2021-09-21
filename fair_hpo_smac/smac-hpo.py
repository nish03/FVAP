#!/usr/bin/env python
import logging
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from os import makedirs, path
from sys import argv

import numpy.random
import torch.random
from numpy.random import RandomState
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario
from torch import cuda, float32, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from data.CelebA import CelebADataset, load_celeba
from data.UTKFace import UTKFaceDataset, load_utkface
from hpo.Cost import cost_functions
from hpo.Hyperparameters import hyperparameters_from_config
from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder

start_date = datetime.now()

logging.basicConfig(level=logging.DEBUG, force=True)

arg_parser = ArgumentParser(
    description="Perform HPO with SMAC to train a generative model"
)
dataset_dir_group = arg_parser.add_mutually_exclusive_group(required=True)
arg_parser.add_argument(
    "--batch-size",
    default=144,
    type=int,
    required=False,
    help="Batch size used for loading the dataset",
)
dataset_dir_group.add_argument(
    "--celeba-dir",
    help="CelebA dataset directory",
    required=False,
),
arg_parser.add_argument(
    "--cost",
    default="MS-SSIM",
    choices=["MS-SSIM", "FairMS-SSIM"],
    required=False,
    help="Cost function used for HPO",
)
arg_parser.add_argument(
    "--datasplit-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for creating random train, validation and tast dataset splits",
)
arg_parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    required=False,
    help="Number of epochs used for training the generative model",
)
arg_parser.add_argument(
    "--image-size",
    default=64,
    type=int,
    required=False,
    help="Image size used for loading the dataset",
)
arg_parser.add_argument(
    "--in-memory-dataset",
    action="store_true",
    help="Load dataset into memory",
)
arg_parser.add_argument(
    "--output-dir",
    default=".",
    required=False,
    help="Directory for log files, save states and HPO output",
)
arg_parser.add_argument(
    "--sensitive-attribute",
    type=int,
    default=0,
    required=False,
    help="Index of the sensitive attribute",
)
arg_parser.add_argument(
    "--smac-initial-design",
    default="Sobol",
    choices=["DefaultConfiguration", "Sobol"],
    required=False,
    help="Initial design for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-pcs-file",
    required=True,
    help="Parameter configuration file used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-runcount",
    default=None,
    type=int,
    required=False,
    help="Maximum number of runs during hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-runtime",
    default=None,
    type=int,
    required=False,
    help="Maximum amount runtime used for hyperparameter optimization with SMAC",
)
arg_parser.add_argument(
    "--smac-seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for hyperparameter optimization with SMAC",
)
dataset_dir_group.add_argument(
    "--utkface-dir",
    help="UTKFace dataset directory",
    required=False,
),
args = arg_parser.parse_args(argv[1:])

opt_params = deepcopy(args)
del opt_params.in_memory_dataset

output_directory = path.join(
    opt_params.output_dir, start_date.strftime("%Y-%m-%d_%H:%M:%S_%f")
)

makedirs(output_directory, exist_ok=True)
log_file_path = path.join(output_directory, "log.txt")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, force=True)
print(f"Logging started in output directory {output_directory}")
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
    cudnn.deterministic = True
    cudnn.benchmark = False
logging.info(
    f"CUDNN convolution benchmarking was {'enabled' if cudnn.benchmark else 'disabled'}"
)

logging.info(
    f"Data will be loaded from {'memory' if args.in_memory_dataset else 'disk'} with "
    f"sensitive attribute {opt_params.sensitive_attribute}, "
    f"image size {opt_params.image_size}, batch size {opt_params.batch_size} and "
    f"datasplit seed {opt_params.datasplit_seed}"
)
logging.info(
    f"Generative model will be trained for {opt_params.epochs} epochs and "
    f"batch size {opt_params.batch_size}"
)
logging.info(
    f"Hyperparameter optimisation with SMAC will be run with "
    f"{'infinite' if opt_params.smac_runtime is None else opt_params.smac_time + 's'}"
    " runtime limit, "
    f"{'infinite' if opt_params.smac_runcount is None else opt_params.smac_runcount}"
    f" runcount limit, parameter configuration file '{opt_params.smac_pcs_file}', "
    f"{opt_params.smac_initial_design} initial design, seed {opt_params.smac_seed} "
    f"and cost function {opt_params.cost}"
)

num_workers = device_count * 4
transform = Compose(
    [
        ConvertImageDtype(float32),
        Resize(opt_params.image_size),
        Lambda(lambda x: 2 * x - 1),
    ]
)
target_transform = Lambda(lambda x: x[opt_params.sensitive_attribute])

if opt_params.utkface_dir is not None:
    dataset_directory = opt_params.utkface_dir
    dataset_class = UTKFaceDataset
    train_dataset, validation_dataset, _ = load_utkface(
        random_split_seed=opt_params.datasplit_seed,
        image_directory_path=dataset_directory,
        transform=transform,
        target_transform=target_transform,
        in_memory=args.in_memory_dataset,
    )
elif opt_params.celeba_dir is not None:
    dataset_directory = opt_params.celeba_dir
    dataset_class = CelebADataset
    train_dataset, validation_dataset, _ = load_celeba(
        image_directory_path=dataset_directory,
        transform=transform,
        target_transform=target_transform,
        in_memory=args.in_memory_dataset,
    )
else:
    raise RuntimeError("No dataset directory was specified")

if not (0 <= opt_params.sensitive_attribute < len(dataset_class.target_attributes)):
    raise RuntimeError("Sensitive attribute index is out of range")

sensitive_attribute = dataset_class.target_attributes[opt_params.sensitive_attribute]

train_dataloader, validation_dataloader = [
    DataLoader(
        dataset,
        batch_size=opt_params.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    for dataset in [train_dataset, validation_dataset]
]

logging.info(
    f"{dataset_class.__name__} was loaded from directory '{dataset_directory}' "
    f"with target sensitive attribute {sensitive_attribute.__name__}"
)

save_file_directory = path.join(output_directory, "save-states")
makedirs(save_file_directory)
optimization_parameters_save_file_path = path.join(save_file_directory, "opt-params.pt")
save(opt_params, optimization_parameters_save_file_path)


def hyperparameter_cost(hyperparameter_config, seed):
    hyperparameter_cost.run += 1

    hyperparameter_cost.seed = seed
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)

    hyper_params = hyperparameters_from_config(
        hyperparameter_config, max_iteration=opt_params.epochs * len(train_dataloader)
    )
    hyperparameter_cost.hyper_params = deepcopy(hyper_params)

    def train_criterion(_model, _data, _target, _output, _mu, _log_var, _data_fraction):
        if isinstance(_model, DataParallel):
            _model = _model.module
        train_criterion.iteration += 1
        return _model.criterion(
            _data,
            _target,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )

    train_criterion.iteration = 0

    def validation_criterion(
        _model, _data, _target, _output, _mu, _log_var, _data_fraction
    ):
        if isinstance(_model, DataParallel):
            _model = _model.module
        return _model.criterion(
            _data,
            _target,
            _output,
            _mu,
            _log_var,
            train_criterion.iteration,
            _data_fraction,
        )

    model = FlexVAE(
        opt_params.image_size,
        hyper_params.latent_dimension_count,
        hyper_params.hidden_layer_count,
        hyper_params.vae_loss_gamma,
        hyper_params.C_max,
        hyper_params.C_stop_iteration,
        hyper_params.reconstruction_loss,
        hyper_params.reconstruction_loss_args,
        hyper_params.reconstruction_loss_label_weights,
        hyper_params.kld_loss_label_weights,
        hyper_params.weighted_average_type,
    )
    if device_count > 1:
        model = DataParallel(model)
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=hyper_params.learning_rate,
        weight_decay=hyper_params.weight_decay,
    )
    lr_scheduler = ExponentialLR(optimizer, gamma=hyper_params.lr_scheduler_gamma)

    train_epoch_losses, validation_epoch_losses = train_variational_autoencoder(
        model,
        optimizer,
        lr_scheduler,
        opt_params.epochs,
        train_criterion,
        validation_criterion,
        train_dataloader,
        validation_dataloader,
        schedule_lr_after_epoch=True,
        display_progress=False,
    )

    if isinstance(model, DataParallel):
        model = model.module
    cost_value, additional_info = hyperparameter_cost.function(
        model, validation_dataloader, sensitive_attribute
    )

    model_state = {
        "additional_info": additional_info,
        "cost_value": cost_value,
        "hyper_params": hyperparameter_cost.hyper_params,
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run": hyperparameter_cost.run,
        "seed": hyperparameter_cost.seed,
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_save_file_name = f"model-run-{hyperparameter_cost.run:04}.pt"
    model_save_file_path = path.join(save_file_directory, model_save_file_name)
    save(model_state, model_save_file_path)
    return cost_value, additional_info


hyperparameter_cost.run = 0
hyperparameter_cost.seed = None
hyperparameter_cost.hyper_params = None
hyperparameter_cost.function = cost_functions[opt_params.cost]


smac_output_directory = path.join(output_directory, "smac-output")
scenario_dict = {
    "pcs_fn": opt_params.smac_pcs_file,
    "run_obj": "quality",
    "deterministic": "false",
    "limit_resources": False,
    "output_dir": smac_output_directory,
    "abort_on_first_run_crash": False,
}
if opt_params.smac_runcount is not None:
    scenario_dict["ta_run_limit"] = opt_params.smac_runcount
if opt_params.smac_runtime is not None:
    scenario_dict["wallclock-limit"] = opt_params.smac_runtime
scenario = Scenario(scenario_dict)
initial_designs = {"DefaultConfiguration": DefaultConfiguration, "Sobol": SobolDesign}
smac = SMAC4HPO(
    scenario=scenario,
    rng=RandomState(opt_params.smac_seed),
    tae_runner=hyperparameter_cost,
    initial_design=initial_designs[opt_params.smac_initial_design],
    intensifier_kwargs={"maxR": 5},
    run_id=1,
)
incumbent_hyperparameter_config = smac.optimize()
run_history = smac.get_runhistory()
trajectory = smac.get_trajectory()
logging.info(f"SMAC HPO finished with Incumbent {incumbent_hyperparameter_config}")

smac_results_state = {
    "scenario": scenario,
    "incumbent_hyperparameter_config": incumbent_hyperparameter_config,
    "run_history": run_history,
    "trajectory": trajectory,
}
smac_results_save_file_path = path.join(save_file_directory, "smac-results.pt")
save(smac_results_state, smac_results_save_file_path)

end_date = datetime.now()
duration = end_date - start_date
logging.info(
    f"Script finished at {end_date} with a runtime of {duration.total_seconds()}"
)
