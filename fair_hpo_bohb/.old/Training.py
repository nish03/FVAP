from tqdm import tqdm

def train_classifier(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    criterion,
    binary=False,
    display_progress=True,
):
    model.train()
    device = next(model.parameters()).device

    final_loss = 0
    correct_prediction_count = 0
    processed_prediction_count = 0
    data_iterator = tqdm(dataloader, leave=False) if display_progress else dataloader
    for data, target in data_iterator:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        if binary:
            output = output.view_as(target)
        loss = criterion(output, target.to(output.dtype))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_value = loss.item()
        final_loss += loss_value
        if binary:
            prediction = output >= 0.5
        else:
            prediction = output.argmax(dim=1, keepdim=True)
        correct_prediction_count += (
            prediction.eq(target.view_as(prediction)).sum().item()
        )
        processed_prediction_count += len(data)

        accuracy = correct_prediction_count / processed_prediction_count
        if display_progress:
            data_iterator.set_description(
                desc=f"Loss={loss_value} Accuracy={100 * accuracy:0.2f}"
            )

    final_loss /= len(dataloader)
    final_accuracy = correct_prediction_count / processed_prediction_count
    return final_loss, final_accuracy
