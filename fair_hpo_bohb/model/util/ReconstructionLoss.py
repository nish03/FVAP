from math import ceil, exp

from torch import cosh, log, stack, tensor
from torch.nn import Module, L1Loss, MSELoss
from torch.nn.functional import avg_pool2d, conv2d, relu


class LogCoshLoss(Module):
    def __init__(self, a=10.0):
        super(LogCoshLoss, self).__init__()
        self.a = a

    def forward(self, reconstruction, data):
        return log(cosh(self.a * (reconstruction - data))).mean() / self.a


class MultiScaleSSIMLoss(Module):
    def __init__(self, window_sigma=1.5, reduction="mean"):
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/mssim_vae.py
            (Apache-2.0 License)
        :param window_sigma: (float)
        :param reduction: (bool)
        """
        super(MultiScaleSSIMLoss, self).__init__()
        self.window_sigma = window_sigma
        self.window_size = 2 * ceil(3.0 * window_sigma) + 1
        self.reduction = reduction

    @staticmethod
    def gaussian_window(window_size, sigma):
        kernel = tensor(
            [
                exp((x - window_size // 2) ** 2 / (2 * sigma ** 2))
                for x in range(window_size)
            ]
        )
        return kernel / kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, self.window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            in_channels, 1, window_size, window_size
        ).contiguous()
        return window

    def ssim(self, img1, img2, window_size):
        device = "cpu" #img1.device
        window = self.create_window(window_size, 3).to(device)
        mu1 = conv2d(img1, window, padding=window_size // 2, groups=3)
        mu2 = conv2d(img2, window, padding=window_size // 2, groups=3)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            conv2d(img1 * img1, window, padding=window_size // 2, groups=3) - mu1_sq
        )
        sigma2_sq = (
            conv2d(img2 * img2, window, padding=window_size // 2, groups=3) - mu2_sq
        )
        sigma12 = (
            conv2d(img1 * img2, window, padding=window_size // 2, groups=3) - mu1_mu2
        )

        img_range = img1.max() - img1.min()
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map

        ssim = ssim_map.flatten(2).mean(-1)
        cs = cs_map.flatten(2).mean(-1)

        return ssim, cs

    def forward(self, img1, img2):
        device = "cpu" #img1.device
        weights = tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mcs = []

        ssim = None
        for level in range(levels):
            ssim, cs = self.ssim(img1, img2, self.window_size)

            if level < levels - 1:
                mcs.append(relu(cs))
                img1 = avg_pool2d(img1, (2, 2))
                img2 = avg_pool2d(img2, (2, 2))

        ssim = relu(ssim)
        mcs_and_ssim = stack(mcs + [ssim])

        ms_ssim = (mcs_and_ssim ** weights.view(-1, 1, 1)).prod(0).mean(-1)
        ms_ssim_loss = 1.0 - ms_ssim

        if self.reduction == "mean":
            return ms_ssim_loss.mean()
        elif self.reduction == "sum":
            return ms_ssim_loss.sum()
        return ms_ssim_loss


reconstruction_losses = {
    "MAE": L1Loss,
    "MSE": MSELoss,
    "MS-SSIM": MultiScaleSSIMLoss,
    "LogCosh": LogCoshLoss,
}
