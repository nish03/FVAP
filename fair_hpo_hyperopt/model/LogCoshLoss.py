from torch import cosh, log
from torch.nn import Module


class LogCoshLoss(Module):
    def __init__(self, a=10.0, reduction="mean"):
        super(LogCoshLoss, self).__init__()
        self.a = a
        self.reduction = reduction

    def forward(self, reconstruction, data):
        log_cosh_loss = log(cosh(self.a * (reconstruction - data))) / self.a
        if self.reduction == "mean":
            return log_cosh_loss.mean()
        elif self.reduction == "sum":
            return log_cosh_loss.sum()
        return log_cosh_loss
