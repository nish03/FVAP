from torch.nn import Module
from torch import log, cosh


class LogCoshLoss(Module):
    def __init__(self, a=10.0):
        super(LogCoshLoss, self).__init__()
        self.a = a

    def forward(self, reconstruction, data):
        return log(cosh(self.a * (reconstruction - data))).mean() / self.a
