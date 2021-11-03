import logging
from collections import defaultdict

from torch import no_grad, stack
from tqdm import tqdm


def train_variational_autoencoder(
    model,
    optimizer,
    lr_scheduler,
    epoch_count,
    train_criterion,
    validation_criterion,
    train_dataloader,
    validation_dataloader,
    schedule_lr_after_epoch=True,
    display_progress=True,
):
    train_epoch_losses = defaultdict(list)
    validation_epoch_losses = defaultdict(list)

    for epoch in range(1, epoch_count + 1):
        logging.debug(f"  Epoch: {epoch}")

        train_losses = train_variational_autoencoder_epoch(
            model,
            optimizer,
            lr_scheduler,
            schedule_lr_after_epoch,
            train_criterion,
            train_dataloader,
            display_progress=display_progress,
        )
        logging.debug(
            "    Training Losses - "
            + " ".join([f"{name}: {value}" for name, value in train_losses.items()])
        )

        validation_losses = evaluate_variational_autoencoder_epoch(
            model, validation_criterion, validation_dataloader
        )
        logging.debug(
            "    Validation Losses - "
            + " ".join(
                [f"{name}: {value}" for name, value in validation_losses.items()]
            )
        )

        nan_loss_encountered = False
        for name, value in train_losses.items():
            if value.isnan().any():
                nan_loss_encountered = True
            train_epoch_losses[name].append(value)
        for name, value in validation_losses.items():
            if value.isnan().any():
                nan_loss_encountered = True
            validation_epoch_losses[name].append(value)

        if nan_loss_encountered:
            logging.debug(f"    Encountered NaN loss value, aborting training")
            break

    for name, loss_values in train_epoch_losses.items():
        train_epoch_losses[name] = stack(loss_values, -1).tolist()

    for name, loss_values in validation_epoch_losses.items():
        validation_epoch_losses[name] = stack(loss_values, -1).tolist()

    return train_epoch_losses, validation_epoch_losses


def train_variational_autoencoder_epoch(
    model,
    optimizer,
    lr_scheduler,
    schedule_lr_after_epoch,
    criterion,
    dataloader,
    display_progress=True,
):
    model.train()
    device = next(model.parameters()).device

    mean_losses = {}
    data_iterator = tqdm(dataloader, leave=False) if display_progress else dataloader
    for data, target in data_iterator:
        data, target = data.to(device), target.to(device)

        data_fraction = len(data) / len(dataloader.dataset)

        output, _, mu, log_var = model(data)
        losses = criterion(model, data, target, output, mu, log_var, data_fraction)

        optimizer.zero_grad(set_to_none=True)

        optimization_loss = losses["ELBO"]
        optimization_loss.backward()

        optimizer.step()

        for name, loss in losses.items():
            if name in mean_losses:
                mean_losses[name] += loss
            else:
                mean_losses[name] = loss

        if display_progress:
            description = "Training Losses - " + " ".join(
                [
                    f"{name}: {loss.item():0.4f}"
                    for name, loss in losses.items()
                    if loss.dim() == 0
                ]
            )
            data_iterator.set_description(desc=description)
        if not schedule_lr_after_epoch and lr_scheduler is not None:
            lr_scheduler.step()

    if schedule_lr_after_epoch and lr_scheduler is not None:
        lr_scheduler.step()

    for name in mean_losses:
        mean_losses[name] /= len(dataloader)
    return mean_losses


def evaluate_variational_autoencoder_epoch(model, criterion, dataloader):
    model.eval()
    device = next(model.parameters()).device

    mean_losses = {}
    with no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            data_fraction = len(data) / len(dataloader.dataset)
            output, _, mu, log_var = model(data)
            losses = criterion(model, data, target, output, mu, log_var, data_fraction)
            for name, loss in losses.items():
                if name in mean_losses:
                    mean_losses[name] += loss
                else:
                    mean_losses[name] = loss

    for name in mean_losses:
        mean_losses[name] /= len(dataloader)
    return mean_losses
