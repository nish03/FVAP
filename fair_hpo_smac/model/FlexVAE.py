from numpy import clip, prod
from torch import flatten, randn, randn_like, cat, zeros, tensor
from torch.nn import (
    BatchNorm2d,
    Parameter,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Linear,
    Module,
    MSELoss,
    Sequential,
    Tanh,
)


class FlexVAE(Module):
    def __init__(
        self,
        image_size,
        latent_dimension_count=128,
        hidden_layer_count=5,
        gamma=10.0,
        c_max=25.0,
        c_stop_iteration=10000,
        reconstruction_loss=MSELoss,
        reconstruction_loss_args=None,
        reconstruction_loss_label_weights=None,
        kld_loss_label_weights=None,
        weighted_average_type=None,
    ):
        super(FlexVAE, self).__init__()

        self.gamma = gamma
        self.c_max = c_max
        self.c_stop_iteration = c_stop_iteration
        self.latent_dimension_count = latent_dimension_count

        # encoder
        encoder_layers = []
        self.encoder = []
        in_channel_count = 3
        hidden_channel_counts = [32 * 2 ** i for i in range(hidden_layer_count)]
        self.max_hidden_channel_count = hidden_channel_counts[-1]
        self.min_hidden_channel_count = hidden_channel_counts[0]
        for out_channel_count in hidden_channel_counts:
            hidden_layer = Sequential(
                Conv2d(
                    in_channel_count,
                    out_channel_count,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                BatchNorm2d(out_channel_count),
                LeakyReLU(inplace=True),
            )
            encoder_layers.append(hidden_layer)
            in_channel_count = out_channel_count

        self.encoder = Sequential(*encoder_layers)

        test_input = zeros(1, 3, image_size, image_size)
        test_encoder_output = self.encoder(test_input)
        self.encoder_output_image_size = test_encoder_output.shape[2:4]
        encoder_output_dims = self.max_hidden_channel_count * prod(
            self.encoder_output_image_size
        )

        self.fc_mu = Linear(encoder_output_dims, self.latent_dimension_count)
        self.fc_var = Linear(encoder_output_dims, self.latent_dimension_count)

        # decoder
        self.decoder_input = Linear(self.latent_dimension_count, encoder_output_dims)

        decoder_layers = []
        hidden_channel_counts.reverse()
        for i in range(hidden_layer_count - 1):
            in_channel_count = hidden_channel_counts[i]
            out_channel_count = hidden_channel_counts[i + 1]
            hidden_decoder_layer = Sequential(
                ConvTranspose2d(
                    in_channel_count,
                    out_channel_count,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=(1, 1),
                    bias=False,
                ),
                BatchNorm2d(out_channel_count),
                LeakyReLU(inplace=True),
            )
            decoder_layers.append(hidden_decoder_layer)
        final_hidden_decoder_layer = Sequential(
            ConvTranspose2d(
                self.min_hidden_channel_count,
                self.min_hidden_channel_count,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            ),
            BatchNorm2d(self.min_hidden_channel_count),
            LeakyReLU(inplace=True),
            Conv2d(
                self.min_hidden_channel_count, 3, kernel_size=(3, 3), padding=(1, 1)
            ),
            Tanh(),
        )
        decoder_layers.append(final_hidden_decoder_layer)
        self.decoder = Sequential(*decoder_layers)

        reconstruction_loss_args = (
            {} if reconstruction_loss_args is None else reconstruction_loss_args
        )
        reconstruction_loss_args["reduction"] = "none"
        self.reconstruction_criterion = reconstruction_loss(**reconstruction_loss_args)

        if weighted_average_type == "IndividualLosses":
            self._reconstruction_loss_fn = self._reconstruction_individual_losses
            self._kld_loss_fn = self._kld_individual_losses
            self._weighted_average_loss_fn = self._weighted_average_individual_losses
        else:
            self._reconstruction_loss_fn = self._reconstruction_batch_loss
            self._kld_loss_fn = self._kld_batch_loss
            if weighted_average_type == "BatchLosses":
                self._weighted_average_loss_fn = self._weighted_average_batch_losses
            else:
                self._weighted_average_loss_fn = self._batch_loss

        if reconstruction_loss_label_weights is not None:
            self.register_parameter(
                "reconstruction_weights",
                Parameter(
                    tensor(reconstruction_loss_label_weights), requires_grad=False
                ),
            )
        else:
            self.reconstruction_weights = None

        if kld_loss_label_weights is not None:
            self.register_parameter(
                "kld_weights",
                Parameter(tensor(kld_loss_label_weights), requires_grad=False),
            )
        else:
            self.kld_weights = None

    def encode(self, x):
        y = self.encoder(x)
        y = flatten(y, start_dim=1)
        mu = self.fc_mu(y)
        log_var = self.fc_var(y)

        return mu, log_var

    def decode(self, z):
        y = self.decoder_input(z)
        y = y.view(
            -1,
            self.max_hidden_channel_count,
            *self.encoder_output_image_size,
        )
        y = self.decoder(y)
        return y

    @staticmethod
    def reparameterize(mu, log_var):
        std = (0.5 * log_var).exp()
        eps = randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y = self.decode(z)
        return y, mu, log_var

    def _reconstruction_batch_loss(self, x, y):
        # reconstruction criterion performs mean reduction
        return self.reconstruction_criterion((y + 1.0) / 2.0, (x + 1.0 / 2.0)).mean()

    def _reconstruction_individual_losses(self, x, y):
        losses = self.reconstruction_criterion((y + 1.0) / 2.0, (x + 1.0 / 2.0))
        if losses.dim() > 1:
            return losses.mean(dim=[i for i in range(1, losses.dim())])
        return losses

    @staticmethod
    def _kld_batch_loss(mu, log_var):
        return (-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1)).mean()

    @staticmethod
    def _kld_individual_losses(mu, log_var):
        return -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1)

    @staticmethod
    def _batch_loss(batch_loss, args, *_):
        return batch_loss(*args)

    @staticmethod
    def _weighted_average_individual_losses(individual_losses, args, labels, weights):
        return (
            cat(
                [
                    individual_losses(*[arg[mask] for arg in args]) * weights[label]
                    for label in range(len(weights))
                    if (mask := labels == label).sum() > 0
                ]
            ).sum()
            / (
                weights[(contained_labels := labels.unique(return_counts=True))[0]]
                * contained_labels[1]
            ).sum()
        )

    @staticmethod
    def _weighted_average_batch_losses(batch_loss, args, labels, weights):
        return (
            cat(
                [
                    batch_loss(*[arg[mask] for arg in args]).view(1) * weights[label]
                    for label in range(len(weights))
                    if (mask := labels == label).sum() > 0
                ]
            ).sum()
            / weights[labels.unique()].sum()
        )

    def criterion(self, x, labels, y, mu, log_var, iteration, kld_weight):
        reconstruction_loss = self._weighted_average_loss_fn(
            self._reconstruction_loss_fn,
            [x, y],
            labels,
            self.reconstruction_weights,
        )
        kld_loss = self._weighted_average_loss_fn(
            self._kld_loss_fn, [mu, log_var], labels, self.kld_weights
        )
        C = clip(self.c_max / self.c_stop_iteration * iteration, 0, self.c_max)
        elbo_loss = reconstruction_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        return {
            "ELBO": elbo_loss,
            "Reconstruction": reconstruction_loss,
            "KLD": kld_loss,
        }

    def sample(self, num_samples, device):
        z = randn(num_samples, self.latent_dimension_count).to(device)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x):
        y = self.forward(x)[0]
        return y
