from tqdm import tqdm
from collections import defaultdict


def train_variational_autoencoder(
    model, dataloader, optimizer, lr_scheduler, criterion, display_progress=True
):
    model.train()
    device = next(model.parameters()).device

    final_losses = defaultdict(float)
    data_iterator = tqdm(dataloader, leave=False) if display_progress else dataloader
    for data, target in data_iterator:
        data, target = data.to(device), target.to(device)

        data_fraction = len(data) / len(dataloader.dataset)

        output, mu, log_var = model(data)
        losses = criterion(data, target, output, mu, log_var, data_fraction)

        optimizer.zero_grad(set_to_none=True)

        optimization_loss = losses["ELBO"]
        optimization_loss.backward()

        optimizer.step()

        for name, loss in losses.items():
            final_losses[name] += loss.item()

        if display_progress:
            description = "Losses - " + " ".join([f"{name}: {loss.item()}" for name, loss in losses.items()])
            data_iterator.set_description(desc=description)

    lr_scheduler.step()

    for name in final_losses:
        final_losses[name] /= len(dataloader)
    return final_losses
