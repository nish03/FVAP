from collections import namedtuple
from itertools import count

from numpy import array, log

from model.util.ReconstructionLoss import reconstruction_losses

hyperparameter_names = [
    "C_max",  # FlexVAE disentangled beta VAE KLD-Loss
    "C_stop_iteration",  # FlexVAE disentangled beta VAE KLD-Loss
    # NOTE: from C_stop_fraction
    "hidden_layer_count",  # FlexVAE network architecture
    "kld_loss_label_weights",  # FlexVAE weighted average KLD loss
    # NOTE: from kld_loss_label_weight_*
    "latent_dimension_count",  # FlexVAE network architecture
    "learning_rate",  # Adam optimizer
    "lr_scheduler_gamma",  # ExponentialLR learning rate scheduler
    "reconstruction_loss",  # FlexVAE reconstruction loss
    "reconstruction_loss_args",  # FlexVAE reconstruction loss
    # NOTE: from ms_ssim_window_sigma and logcosh_a
    "reconstruction_loss_label_weights",  # FlexVAE weighted average reconstruction loss
    # NOTE: from reconstruction kld_loss_label_weight_{LABEL}
    "vae_loss_gamma",  # FlexVAE KLD-Loss disentangled beta VAE
    "weight_decay",  # Adam optimizer
    "weighted_average_type",  # FlexVAE weighted average loss
]

Hyperparameters = namedtuple("Hyperparameters", hyperparameter_names)


def hyperparameters_from_config(hyperparameter_config, max_iteration):
    params = dict(**hyperparameter_config)

    params["C_stop_iteration"] = params.pop("C_stop_fraction") * max_iteration

    reconstruction_loss_args = {}
    reconstruction_loss_name = params["reconstruction_loss"]
    if reconstruction_loss_name == "MS-SSIM":
        reconstruction_loss_args["window_sigma"] = params.pop("ms_ssim_window_sigma")
    elif reconstruction_loss_name == "LogCosh":
        reconstruction_loss_args["a"] = params.pop("logcosh_a")
    params["reconstruction_loss_args"] = reconstruction_loss_args

    params["reconstruction_loss"] = reconstruction_losses[reconstruction_loss_name]

    reconstruction_label_weights = []
    for label in count():
        weight_name = f"reconstruction_loss_label_weight_{label}"
        if weight_name in params:
            reconstruction_label_weights.append(params.pop(weight_name))
        else:
            break
    if len(reconstruction_label_weights) > 0:
        # generate convex weight combination from uniformly sampled weights,
        # by transformation to normalized exponential distribution
        # see https://cs.stackexchange.com/q/3229
        reconstruction_label_weights = -log(array(reconstruction_label_weights))
        reconstruction_label_weights /= reconstruction_label_weights.sum()
        params["reconstruction_loss_label_weights"] = reconstruction_label_weights
    else:
        params["reconstruction_loss_label_weights"] = None

    kld_label_weights = []
    for label in count():
        weight_name = f"kld_loss_label_weight_{label}"
        if weight_name in params:
            kld_label_weights.append(params.pop(weight_name))
        else:
            break
    if len(kld_label_weights) > 0:
        # generate convex weight combination from uniformly sampled weights,
        # by transformation to normalized exponential distribution
        # see https://cs.stackexchange.com/q/3229
        kld_label_weights = -log(array(kld_label_weights))
        kld_label_weights /= kld_label_weights.sum()
        params["kld_loss_label_weights"] = kld_label_weights
    else:
        params["kld_loss_label_weights"] = None

    if "weighted_average_type" not in params:
        params["weighted_average_type"] = "None"

    hyperparameters = Hyperparameters(**params)
    return hyperparameters
