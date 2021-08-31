from torch import stack, tensor
from math import exp
from torch.nn import Module
from torch.nn.functional import avg_pool2d, conv2d


class MultiScaleSSIMLoss(Module):
    def __init__(self, window_size=11):
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/mssim_vae.py
            (Apache-2.0 License)
        :param window_size: (Int)
        """
        super(MultiScaleSSIMLoss, self).__init__()
        self.window_size = window_size

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
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            in_channels, 1, window_size, window_size
        ).contiguous()
        return window

    def ssim(self, img1, img2, window_size, in_channel):
        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = conv2d(img1, window, padding=window_size // 2, groups=in_channel)
        mu2 = conv2d(img2, window, padding=window_size // 2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            conv2d(img1 * img1, window, padding=window_size // 2, groups=in_channel)
            - mu1_sq
        )
        sigma2_sq = (
            conv2d(img2 * img2, window, padding=window_size // 2, groups=in_channel)
            - mu2_sq
        )
        sigma12 = (
            conv2d(img1 * img2, window, padding=window_size // 2, groups=in_channel)
            - mu1_mu2
        )

        img_range = img1.max() - img1.min()
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = (v1 / v2).mean()

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        ret = ssim_map.mean()
        return ret, cs

    def forward(self, img1, img2):
        device = img1.device
        weights = tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2, self.window_size, 3)
            mssim.append(sim)
            mcs.append(cs)

            img1 = avg_pool2d(img1, (2, 2))
            img2 = avg_pool2d(img2, (2, 2))

        mssim = stack(mssim)
        mcs = stack(mcs)

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = (pow1[:-1] * pow2[-1]).prod()
        return 1 - output
