import comet_ml
import torch.utils.data

from Evaluation import evaluate
from Util import get_device
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy


# def intersection_over_union(
#     sensitive_class_index,
#     sensitive_labels,
#     target_class_index,
#     target_probabilities,
#     target_labels,
# ):
#     sensitive_samples = sensitive_labels == sensitive_class_index
#     confusion_matrix = (
#         target_probabilities[sensitive_samples].unsqueeze(dim=2)
#         * one_hot(
#             target_labels[sensitive_samples], num_classes=target_probabilities.shape[1]
#         ).unsqueeze(dim=1)
#     ).sum(dim=0)
#     return confusion_matrix[target_class_index, target_class_index] / (
#         confusion_matrix[target_class_index, :].sum()
#         + confusion_matrix[:, target_class_index].sum()
#         - confusion_matrix[target_class_index, target_class_index]
#     )

# sensitive attribute a attribute (age)
# a=0 ~ younger than 60 years
# a=1 ~ 60 years and older,
# sensitive_attribute_index = 0
# sensitive_attribute_class_count = train_dataset.attribute_class_counts[
#     sensitive_attribute_index
# ]
# sensitive_attribute_labels = labels[:, sensitive_attribute_index]

# target attribute y (gender)
# y=0 ~ female
# y=1 ~ male
# target_attribute_index = 1
# target_attribute_class_count = train_dataset.attribute_class_counts[
#     target_attribute_index
# ]
# target_attribute_probabilities = class_probabilities(outputs, target_attribute_index)
# target_attribute_labels = labels[:, target_attribute_index]

# iou = tensor(
#     [
#         [
#             intersection_over_union(
#                 sensitive_attribute_class_index,
#                 sensitive_attribute_labels,
#                 target_attribute_class_index,
#                 target_attribute_probabilities,
#                 target_attribute_labels,
#             )
#             for sensitive_attribute_class_index in range(
#             sensitive_attribute_class_count
#         )
#         ]
#         for target_attribute_class_index in range(target_attribute_class_count)
#     ]
# )
# iou_sensitive_attribute = iou.mean(dim=0)
# iou_loss_terms = (
#         iou_sensitive_attribute.unsqueeze(dim=1) - iou_sensitive_attribute.unsqueeze(dim=0)
# ).pow(2)
# iou_loss = iou_loss_terms.sum() / (
#         sensitive_attribute_class_count * (sensitive_attribute_class_count - 1)
# )


def criterion(
    model,
    multi_output_class_logits,
    multi_output_class_probabilities,
    attribute_targets,
    sensitive_attribute_index,
    target_attribute_index,
):
    loss_terms = {}
    for class_logits in multi_output_class_logits:
        if class_logits.dim() == 2:
            binary_attribute_targets = attribute_targets[:, model.attribute_sizes.eq(2)]
            loss_terms["binary"] = binary_cross_entropy_with_logits(
                class_logits,
                binary_attribute_targets.float(),
                reduction="sum",
            )
        else:
            attribute_size = class_logits.shape[1]
            categorical_attribute_targets = attribute_targets[
                :, model.attribute_sizes.eq(attribute_size)
            ]
            loss_terms[f"categorical_{attribute_size}"] = cross_entropy(
                class_logits,
                categorical_attribute_targets,
                reduction="sum",
            )
    for loss_part_name in loss_terms:
        loss_terms[loss_part_name] /= (
            attribute_targets.shape[0] * attribute_targets.shape[1]
        )

    loss = sum(loss_terms.values())
    return loss, loss_terms


def train_classifier(
    parallel_model: torch.nn.DataParallel,
    optimizer: torch.optim.Optimizer,
    epoch_count: int,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    experiment: comet_ml.Experiment,
    sensitive_attribute_index: int,
    target_attribute_index: int,
):
    best_model_state = {}
    model = parallel_model.module
    best_valid_loss = None
    device = get_device()
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_eval_state = None
        with experiment.context_manager("train"):
            for data in train_dataloader:
                images, attribute_targets = data[0].to(device), data[1].to(device)

                optimizer.zero_grad(set_to_none=True)

                (
                    multi_output_class_logits,
                    multi_output_class_probabilities,
                    attribute_predictions,
                ) = parallel_model(images)

                loss, loss_terms = criterion(
                    model,
                    multi_output_class_logits,
                    multi_output_class_probabilities,
                    attribute_targets,
                    sensitive_attribute_index,
                    target_attribute_index,
                )
                loss.backward()

                optimizer.step()

                train_eval_results, train_eval_state = evaluate(
                    attribute_predictions,
                    attribute_targets,
                    loss,
                    loss_terms,
                    train_eval_state,
                )
            experiment.log_metrics(train_eval_results, epoch=epoch)

        parallel_model.eval()
        valid_eval_state = None
        with experiment.context_manager("valid"):
            for data in valid_dataloader:
                images, attribute_targets = data[0].to(device), data[1].to(device)

                (
                    multi_output_class_logits,
                    multi_output_class_probabilities,
                    attribute_predictions,
                ) = parallel_model(images)

                loss, loss_terms = criterion(
                    model,
                    multi_output_class_logits,
                    multi_output_class_probabilities,
                    attribute_targets,
                    sensitive_attribute_index,
                    target_attribute_index,
                )

                valid_eval_results, valid_eval_state = evaluate(
                    model,
                    attribute_predictions,
                    attribute_targets,
                    loss,
                    loss_terms,
                    valid_eval_state,
                )

            experiment.log_metrics(valid_eval_results, epoch=epoch)
        epoch_valid_loss = valid_eval_results["loss"]
        if best_valid_loss is None or best_valid_loss < epoch_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_state = {
                "valid_results": valid_eval_results,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

    final_model_state = {
        "valid_results": valid_eval_results,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_count,
    }
    return best_model_state, final_model_state
