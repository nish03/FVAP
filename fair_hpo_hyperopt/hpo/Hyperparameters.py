from collections import namedtuple
from model.util.ReconstructionLoss import reconstruction_losses

hyperparameter_names = [
    "C_max",
    "C_stop_iteration",
    "hidden_layer_count",
    "latent_dimension_count",
    "learning_rate",
    "lr_scheduler_gamma",
    "reconstruction_loss",
    "reconstruction_loss_args",
    "vae_loss_gamma",
    "weight_decay",
]

Hyperparameters = namedtuple("Hyperparameters", hyperparameter_names)


def hyperparameters_from_config(hyperparameter_config, max_iteration):
    hyperparameter_dict = hyperparameter_config
    hyperparameter_dict["C_stop_iteration"] = (
        hyperparameter_dict["C_stop_fraction"] * max_iteration
    )
    del hyperparameter_dict["C_stop_fraction"]
    hyperparameter_dict["reconstruction_loss_args"] = {}
    if hyperparameter_dict["reconstruction_loss"] == "MS-SSIM":
        hyperparameter_dict["reconstruction_loss_args"][
            "window_sigma"
        ] = hyperparameter_dict["ms_ssim_window_sigma"]
        del hyperparameter_dict["ms_ssim_window_sigma"]
    elif hyperparameter_dict["reconstruction_loss"] == "LogCosh":
        hyperparameter_dict["reconstruction_loss_args"]["a"] = hyperparameter_dict[
            "logcosh_a"
        ]
        del hyperparameter_dict["logcosh_a"]
    hyperparameter_dict["reconstruction_loss"] = reconstruction_losses[
        hyperparameter_dict["reconstruction_loss"]
    ]
    hyperparameters = Hyperparameters(**hyperparameter_dict)
    return hyperparameters
