from argparse import ArgumentParser
from copy import deepcopy

from experiment import train_model_experiment
from losses.fair_losses import fair_losses

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a fair attribute prediction model", fromfile_prefix_chars="+")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epoch_count", type=int, default=15, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", default="UTKFace", choices=["UTKFace", "CelebA"], help="Dataset")
    parser.add_argument("--model", default="SlimCNN", choices=["SlimCNN"], help="Prediction model")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam"], help="Optimizer")
    parser.add_argument("--sensitive_attribute_index", required=True, type=int, help="Sensitive attribute index")
    parser.add_argument("--target_attribute_index", required=True, type=int, help="Target attribute index")
    parser.add_argument("--fair_loss_weight", type=float, default=1, help="Fair loss weight")
    parser.add_argument(
        "--fair_loss_type",
        default="IntersectionOverUnion",
        choices=fair_losses.keys(),
        help="Fair loss type",
    )
    parser.add_argument("--experiment_name", default="TrainModel", help="Experiment name")
    parser.set_defaults(offline_experiment=False)
    parser.add_argument("--offline_experiment", action="store_true", help="Offline experiment")

    arguments = parser.parse_args()

    parameters = deepcopy(vars(arguments))
    parameters.pop("experiment_name")
    parameters.pop("offline_experiment")
    experiment_name = arguments.experiment_name
    offline_experiment = arguments.offline_experiment

    train_model_experiment(parameters, experiment_name, offline_experiment)
