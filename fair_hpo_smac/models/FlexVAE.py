from numpy import clip, prod
from torch import flatten, exp, randn_like, mean, sum, randn, zeros
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    ConvTranspose2d,
    Linear,
    Tanh,
)
from torch.nn.functional import mse_loss


class FlexVAE(Module):
    def __init__(self, image_size, latent_dimension_count, hidden_layer_count=5):
        super(FlexVAE, self).__init__()

        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
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
                    bias=False
                ),
                BatchNorm2d(out_channel_count),
                LeakyReLU(inplace=True),
            )
            encoder_layers.append(hidden_layer)
            in_channel_count = out_channel_count

        self.encoder = Sequential(*encoder_layers)
        
        test_input = zeros(1, 3, *image_size)
        test_encoder_output = self.encoder(test_input)
        self.encoder_output_image_size = test_encoder_output.shape[2:4]
        encoder_output_dims = self.max_hidden_channel_count * prod(self.encoder_output_image_size)

        self.fc_mu = Linear(
            encoder_output_dims, self.latent_dimension_count
        )
        self.fc_var = Linear(
            encoder_output_dims, self.latent_dimension_count
        )

        # decoder
        self.decoder_input = Linear(
            self.latent_dimension_count, encoder_output_dims
        )

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
                    bias=False
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
        std = exp(0.5 * log_var)
        eps = randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y = self.decode(z)
        return y, mu, log_var

    @staticmethod
    def criterion(x, y, mu, log_var, gamma, c_max, c_stop_iteration, iteration, kld_weight):
        reconstruction_loss = mse_loss(y, x)
        kld_loss = mean(-0.5 * sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        C = clip(c_max / c_stop_iteration * iteration, 0, c_max)
        elbo_loss = reconstruction_loss + gamma * kld_weight * (kld_loss - C).abs()
        return {"ELBO": elbo_loss, "Reconstruction": reconstruction_loss, "KLD": kld_loss}

    def sample(self, num_samples, device):
        z = randn(num_samples, self.latent_dimension_count).to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        y = self.forward(x)[0]
        return y
