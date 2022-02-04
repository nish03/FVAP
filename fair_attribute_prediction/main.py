#!/usr/bin/env python3

from argparse import ArgumentParser
from copy import deepcopy

from experiment import train_model_experiment
from losses.fair_losses import fair_losses

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a fair attribute prediction model", fromfile_prefix_chars="+")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_count", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning_rate_scheduler",
        default="None",
        choices=["None", "ReduceLROnPlateau"],
    )
    parser.add_argument("--reduce_lr_on_plateau_factor", type=float, default=0.5)
    parser.add_argument("--reduce_lr_on_plateau_patience", type=int, default=5)
    parser.add_argument(
        "--metrics_averaging_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument("--dataset", default="UTKFace", choices=["UTKFace", "CelebA"])
    parser.add_argument("--model", default="SlimCNN", choices=["SlimCNN"])
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sensitive_attribute_index", required=True, type=int)
    parser.add_argument("--target_attribute_index", required=True, type=int)
    parser.add_argument("--fair_loss_weight", type=float, default=1)
    parser.add_argument(
        "--fair_loss_type",
        default="IntersectionOverUnion",
        choices=fair_losses.keys(),
    )
    parser.add_argument("--experiment_name", default="TrainModel")
    parser.set_defaults(offline_experiment=False)
    parser.add_argument("--offline_experiment", action="store_true")
    parser.add_argument("--pretrained_model")

    arguments = parser.parse_args()

    parameters = deepcopy(vars(arguments))
    parameters.pop("experiment_name")
    parameters.pop("offline_experiment")
    experiment_name = arguments.experiment_name
    offline_experiment = arguments.offline_experiment

    train_model_experiment(parameters, experiment_name, offline_experiment)
