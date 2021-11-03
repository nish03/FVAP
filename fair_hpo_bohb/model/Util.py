from torch.nn import L1Loss, MSELoss

from model.LogCoshLoss import LogCoshLoss
from model.MultiScaleSSIMLoss import MultiScaleSSIMLoss

reconstruction_losses = {
    "MAE": L1Loss,
    "MSE": MSELoss,
    "MS-SSIM": MultiScaleSSIMLoss,
    "LogCosh": LogCoshLoss,
}
