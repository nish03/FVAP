#!/usr/bin/env python
import logging
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from os import makedirs, symlink, remove
from os.path import basename, isdir, islink
from os.path import join as join_path
from os.path import relpath
from shutil import copy, copyfileobj, rmtree
from sys import argv

import numpy.random
import torch.random
from numpy.random import RandomState
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.sobol_design import SobolDesign
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from torch import cuda, float32, load, save
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize

from data.CelebA import CelebADataset, load_celeba
from data.UTKFace import UTKFaceDataset, load_utkface
from hpo.Cost import hpo_cost
from hpo.Hyperparameters import hyperparameters_from_config
from model.FlexVAE import FlexVAE
from training.Training import train_variational_autoencoder

start_date = datetime.now()

logging.basicConfig(level=logging.DEBUG, force=True)


class FloatRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __repr__(self):
        return f"[{self.start}, {self.end}]"


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
    "--fair-cost-coefficient",
    default=0.5,
    type=float,
    choices=FloatRange(0.0, 1.0),
    required=False,
    help="Coefficient α in the HPO cost function: (1 - α) * performance + α * fairness",
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
    help="Output directory for log files, save states and HPO runs",
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
    type=int,
    required=True,
    help="Maximum model evaluation runs during hyperparameter optimization with SMAC",
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

resume_experiment = isdir(opt_params.output_dir)
save_states_dir = join_path(opt_params.output_dir, "save-states")
last_opt_params_file_path = join_path(save_states_dir, "opt-params.pt")
last_opt_params = (
    load(last_opt_params_file_path) if islink(last_opt_params_file_path) else None
)
last_smac_results_file_path = join_path(save_states_dir, "smac-results.pt")
last_smac_results = (
    load(last_smac_results_file_path) if islink(last_smac_results_file_path) else None
)
if not resume_experiment:
    makedirs(save_states_dir)
    smac_run = 1
else:
    if last_smac_results is None:
        raise RuntimeError(
            f"Script can't be resumed from exisiting output directory "
            f"{opt_params.output_dir}: results from previous SMAC run are missing!"
        )
    if last_opt_params.smac_runtime is not None and opt_params.smac_runtime is not None:
        opt_params.smac_runtime += last_opt_params.smac_runtime
    opt_params.smac_runcount += last_opt_params.smac_runcount
    smac_run = last_smac_results["smac_run"] + 1


smac_run_dir = join_path(
    opt_params.output_dir,
    f"smac-run-{smac_run:04}",
)
removed_aborted_run = False
if isdir(smac_run_dir):
    rmtree(smac_run_dir)
    removed_aborted_run = True
makedirs(smac_run_dir)
run_log_file_path = join_path(smac_run_dir, "log.txt")
logging.basicConfig(filename=run_log_file_path, level=logging.DEBUG, force=True)
print(f"OutputDirectory={opt_params.output_dir}")
print(f"SMACRun={smac_run}")
logging.info(f"Script {'resumed' if resume_experiment else 'started'} at {start_date}")
if removed_aborted_run:
    logging.info(f"Removed run directory of unfinished SMAC HPO run {smac_run}")

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()
    if not resume_experiment:
        logging.info(
            f"Memory allocation is performed on {device_count} "
            f"CUDA device{'s' if device_count > 1 else ''}"
        )
else:
    device = "cpu"
    device_count = 1
    if not resume_experiment:
        logging.info("Memory allocation is performed on the CPU device")

if cudnn.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False

if not resume_experiment:
    logging.info(
        f"Data will be loaded from {'memory' if args.in_memory_dataset else 'disk'} "
        f"with sensitive attribute {opt_params.sensitive_attribute}, "
        f"image size {opt_params.image_size}, batch size {opt_params.batch_size} and "
        f"datasplit seed {opt_params.datasplit_seed}"
    )
    logging.info(
        f"Generative model will be trained for {opt_params.epochs} epochs and "
        f"batch size {opt_params.batch_size}"
    )
    logging.info(
        f"SMAC HPO run {smac_run} will be started with, "
        f"{'inf' if opt_params.smac_runtime is None else opt_params.smac_runtime}"
        " seconds runtime limit, "
        f"{'inf' if opt_params.smac_runcount is None else opt_params.smac_runcount}"
        f" evaluations runcount limit, "
        f"parameter configuration file '{opt_params.smac_pcs_file}', "
        f"{opt_params.smac_initial_design} initial design, seed {opt_params.smac_seed} "
        f"and fair cost coefficient {opt_params.fair_cost_coefficient}"
    )
else:
    logging.info(
        f"SMAC HPO run {smac_run} will be resumed with "
        f"{'inf' if opt_params.smac_runtime is None else opt_params.smac_runtime}"
        " seconds runtime limit, "
        f"{'inf' if opt_params.smac_runcount is None else opt_params.smac_runcount}"
        f" evaluations runcount limit, "
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
    dataset_dir = opt_params.utkface_dir
    dataset_class = UTKFaceDataset
    train_dataset, validation_dataset, _ = load_utkface(
        random_split_seed=opt_params.datasplit_seed,
        image_dir_path=dataset_dir,
        transform=transform,
        target_transform=target_transform,
        in_memory=args.in_memory_dataset,
    )
elif opt_params.celeba_dir is not None:
    dataset_dir = opt_params.celeba_dir
    dataset_class = CelebADataset
    train_dataset, validation_dataset, _ = load_celeba(
        image_dir_path=dataset_dir,
        transform=transform,
        target_transform=target_transform,
        in_memory=args.in_memory_dataset,
    )
else:
    raise RuntimeError("No dataset directory was specified")

if not (0 <= opt_params.sensitive_attribute < len(dataset_class.target_attributes)):
    raise RuntimeError(
        f"Sensitive attribute index {opt_params.sensitive_attribute} is out of range"
    )

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

if not resume_experiment:
    logging.info(
        f"{dataset_class.__name__} was loaded from directory '{dataset_dir}' "
        f"with target sensitive attribute {sensitive_attribute.__name__}"
    )


def hyperparameter_cost(hyperparameter_config, seed):
    hyperparameter_cost.model_run += 1

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
    cost_value, additional_info = hpo_cost(
        model,
        validation_dataloader,
        sensitive_attribute,
        alpha=opt_params.fair_cost_coefficient,
        window_sigma=0.5,
    )

    model_run_data = {
        "additional_info": additional_info,
        "cost_value": cost_value,
        "hyper_params": hyperparameter_cost.hyper_params,
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_run": hyperparameter_cost.model_run,
        "seed": hyperparameter_cost.seed,
        "train_epoch_losses": train_epoch_losses,
        "validation_epoch_losses": validation_epoch_losses,
    }

    model_run_file_name = f"model-run-{hyperparameter_cost.model_run:04}.pt"
    model_run_file_path = join_path(smac_run_dir, model_run_file_name)
    save(model_run_data, model_run_file_path)
    hyperparameter_cost.model_run_file_paths.append(model_run_file_path)
    return cost_value, additional_info


hyperparameter_cost.model_run = 0
hyperparameter_cost.seed = None
hyperparameter_cost.hyper_params = None
hyperparameter_cost.model_run_file_paths = []

scenario_dict = {
    "pcs_fn": opt_params.smac_pcs_file,
    "run_obj": "quality",
    "deterministic": "false",
    "limit_resources": False,
    "output_dir": smac_run_dir,
    "abort_on_first_run_crash": False,
    "ta_run_limit": opt_params.smac_runcount,
}
if opt_params.smac_runtime is not None:
    scenario_dict["wallclock-limit"] = opt_params.smac_runtime
scenario = Scenario(scenario_dict)

initial_designs = {"DefaultConfiguration": DefaultConfiguration, "Sobol": SobolDesign}
smac_args = {
    "scenario": scenario,
    "tae_runner": hyperparameter_cost,
    "initial_design": initial_designs[opt_params.smac_initial_design],
    "intensifier_kwargs": {"maxR": 5},
    "run_id": 1,
}
if not resume_experiment:
    smac = SMAC4HPO(rng=RandomState(opt_params.smac_seed), **smac_args)
else:
    old_smac_output_dir = join_path(
        opt_params.output_dir,
        f"smac-run-{smac_run - 1:04}",
    )
    old_smac_run_id_dir = join_path(old_smac_output_dir, f"run_{smac_args['run_id']}")
    old_runhistory_json_file_path = join_path(old_smac_run_id_dir, "runhistory.json")
    run_history = RunHistory()
    run_history.load_json(old_runhistory_json_file_path, scenario.cs)
    hyperparameter_cost.model_run = len(run_history.data)
    old_stats_json_file_path = join_path(old_smac_run_id_dir, "stats.json")
    stats = Stats(scenario)
    stats.load(old_stats_json_file_path)
    old_trajectory_json_file_path = join_path(old_smac_run_id_dir, "traj_aclib2.json")
    trajectory = TrajLogger.read_traj_aclib_format(
        fn=old_trajectory_json_file_path, cs=scenario.cs
    )
    incumbent_hyperparameter_config = trajectory[-1]["incumbent"]

    smac_run_id_dir = join_path(
        opt_params.output_dir, f"smac-run-{smac_run:04}", f"run_{smac_args['run_id']}"
    )
    new_trajectory_json_file_path = join_path(smac_run_id_dir, "traj_aclib2.json")
    makedirs(smac_run_id_dir)
    copy(old_trajectory_json_file_path, new_trajectory_json_file_path)

    scenario.output_dir_for_this_run = smac_run_id_dir
    smac = SMAC4HPO(
        runhistory=run_history,
        stats=stats,
        restore_incumbent=incumbent_hyperparameter_config,
        rng=last_smac_results["rng"],
        **smac_args,
    )
    smac.solver.initial_design_configs = last_smac_results[
        "remaining_initial_design_configs"
    ]

    logging.info(f"Restored SMAC state from run directory '{old_smac_run_id_dir}'")


incumbent_hyperparameter_config = smac.optimize()
logging.info(
    f"SMAC HPO run {smac_run} finished with Incumbent {incumbent_hyperparameter_config}"
)
run_history = smac.get_runhistory()
trajectory = smac.get_trajectory()
smac_results_data = {
    "scenario": scenario,
    "incumbent_hyperparameter_config": incumbent_hyperparameter_config,
    "run_history": run_history,
    "trajectory": trajectory,
    "remaining_initial_design_configs": smac.solver.initial_design_configs,
    "rng": smac.solver.rng,
    "smac_run": smac_run,
}
run_smac_results_file_path = join_path(smac_run_dir, "smac-results.pt")
run_opt_params_file_path = join_path(smac_run_dir, "opt-params.pt")
log_file_path = join_path(opt_params.output_dir, "log.txt")
save(opt_params, run_opt_params_file_path)
save(smac_results_data, run_smac_results_file_path)
run_file_paths = [
    run_opt_params_file_path,
    run_smac_results_file_path,
]
run_file_paths += hyperparameter_cost.model_run_file_paths
for run_file_path in run_file_paths:
    link_src_path = relpath(run_file_path, save_states_dir)
    link_dst_path = join_path(save_states_dir, basename(run_file_path))
    if islink(link_dst_path):
        remove(link_dst_path)
    symlink(link_src_path, link_dst_path)


end_date = datetime.now()
duration = end_date - start_date
logging.info(
    f"Script finished at {end_date} with a runtime of {duration.total_seconds()}"
)

if not resume_experiment:
    copy(run_log_file_path, log_file_path)
else:
    with open(log_file_path, "a") as log_file, open(run_log_file_path) as run_log_file:
        copyfileobj(run_log_file, log_file)

print(f"RunCount={len(run_history.data)}")
