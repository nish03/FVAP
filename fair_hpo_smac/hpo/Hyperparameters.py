from collections import namedtuple

Hyperparameters = namedtuple(
    "Hyperparameters",
    [
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
    ],
)
