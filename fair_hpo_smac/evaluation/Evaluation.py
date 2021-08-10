from torch import no_grad
from collections import defaultdict


def evaluate_variational_autoencoder(model, dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device

    mean_losses = defaultdict(float)
    with no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            data_fraction = len(data) / len(dataloader.dataset)
            output, mu, log_var = model(data)
            losses = criterion(data, target, output, mu, log_var, data_fraction)
            for name, loss in losses.items():
                mean_losses[name] += loss.item()

    for name in mean_losses:
        mean_losses[name] /= len(dataloader)
    return mean_losses
