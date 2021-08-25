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
    iteration,
    schedule_lr_after_epoch=True,
    display_progress=True,
):
    train_epoch_losses = defaultdict(list)
    validation_epoch_losses = defaultdict(list)

    for epoch in range(1, epoch_count + 1):
        logging.debug(f"  Epoch: {epoch}")
        print("training")
        train_losses = train_variational_autoencoder_epoch(
            model,
            optimizer,
            lr_scheduler,
            schedule_lr_after_epoch,
            train_criterion,
            train_dataloader,
            iteration,
            display_progress=display_progress,
        )
        logging.debug(
            "    Training Losses - "
            + " ".join([f"{name}: {value}" for name, value in train_losses.items()])
        )
        print("validation")
        validation_losses = evaluate_variational_autoencoder_epoch(
            model, validation_criterion, validation_dataloader, iteration
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
            epoch,
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
    iteration,
    display_progress=True,
):
    model.train()
    device = next(model.parameters()).device

    final_losses = defaultdict(float)
    data_iterator = tqdm(dataloader, leave=False) if display_progress else dataloader
    i = 0
    for data, target in data_iterator:
        data, target = data.to(device), target.to(device)
        i =i+1
        data_fraction = len(data) / len(dataloader.dataset)

        output, mu, log_var = model(data)
        losses = criterion(data,  output, mu, log_var, i, data_fraction)
        #i = i+1
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


def evaluate_variational_autoencoder_epoch(model, criterion, dataloader, iteration):
    model.eval()
    device = next(model.parameters()).device

    mean_losses = defaultdict(float)
    with no_grad():
        i = 0
        for data, target in dataloader:
            i = i+1
            data, target = data.to(device), target.to(device)
            data_fraction = len(data) / len(dataloader.dataset)
            output, mu, log_var = model(data)
            losses = criterion(data, output, mu, log_var, i, data_fraction)
            #i=i+1
            for name, loss in losses.items():
                mean_losses[name] += loss.item()
    for name in mean_losses:
        mean_losses[name] /= len(dataloader)
    return mean_losses
