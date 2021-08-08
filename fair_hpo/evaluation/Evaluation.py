from torch import no_grad
from collections import defaultdict


def evaluate_classifier(model, dataloader, loss_fn, binary=False):
    model.eval()
    device = next(model.parameters()).device

    loss = 0
    correct_prediction_count = 0
    processed_prediction_count = 0
    with no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if binary:
                output = output.view_as(target)
            loss += loss_fn(output, target.to(output.dtype)).item()
            if binary:
                prediction = output >= 0.5
            else:
                prediction = output.argmax(dim=1, keepdim=True)
            correct_prediction_count += (
                prediction.eq(target.view_as(prediction)).sum().item()
            )
            processed_prediction_count += len(data)

    assert(processed_prediction_count > 0)
    accuracy = correct_prediction_count / processed_prediction_count
    return loss, accuracy


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
