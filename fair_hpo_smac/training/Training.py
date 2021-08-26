import logging
from collections import defaultdict

from torch import no_grad
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
    save_model_state_fn,
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

        for name, value in train_losses.items():
            train_epoch_losses[name].append(value)
        for name, value in validation_losses.items():
            validation_epoch_losses[name].append(value)

    save_model_state_fn(
        epoch_count,
        model,
        optimizer,
        lr_scheduler,
        train_epoch_losses,
        validation_epoch_losses,
    )
    return model, train_epoch_losses, validation_epoch_losses


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

    final_losses = defaultdict(float)
    data_iterator = tqdm(dataloader, leave=False) if display_progress else dataloader
    for data, target in data_iterator:
        data, target = data.to(device), target.to(device)

        data_fraction = len(data) / len(dataloader.dataset)

        output, mu, log_var = model(data)
        losses = criterion(model, data, target, output, mu, log_var, data_fraction)

        optimizer.zero_grad(set_to_none=True)

        optimization_loss = losses["ELBO"]
        optimization_loss.backward()

        optimizer.step()

        for name, loss in losses.items():
            final_losses[name] += loss.item()

        if display_progress:
            description = "Training Losses - " + " ".join(
                [f"{name}: {loss.item()}" for name, loss in losses.items()]
            )
            data_iterator.set_description(desc=description)
        if not schedule_lr_after_epoch and lr_scheduler is not None:
            lr_scheduler.step()

    if schedule_lr_after_epoch and lr_scheduler is not None:
        lr_scheduler.step()

    for name in final_losses:
        final_losses[name] /= len(dataloader)
    return final_losses


def evaluate_variational_autoencoder_epoch(model, criterion, dataloader):
    model.eval()
    device = next(model.parameters()).device

    mean_losses = defaultdict(float)
    with no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            data_fraction = len(data) / len(dataloader.dataset)
            output, mu, log_var = model(data)
            losses = criterion(model, data, target, output, mu, log_var, data_fraction)
            for name, loss in losses.items():
                mean_losses[name] += loss.item()

    for name in mean_losses:
        mean_losses[name] /= len(dataloader)
    return mean_losses
