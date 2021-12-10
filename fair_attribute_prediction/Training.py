import comet_ml
import torch.utils.data

from collections import defaultdict
from Evaluation import evaluate
from Util import get_device
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch import stack, softmax, sigmoid, tensor
from torch.nn.functional import one_hot

from MultiAttributeClassifier import MultiAttributeClassifier


def class_probability_predictions(
    model: MultiAttributeClassifier, outputs: list[torch.Tensor], attribute_index: int
):
    attribute_size = model.attribute_sizes[attribute_index]
    previous_attribute_sizes = model.attribute_sizes[0:attribute_index]
    output_prediction_index = tensor(
        previous_attribute_sizes == attribute_size,
        dtype=torch.int64,
    ).sum()
    output_index = model.inverse_attribute_size_indices[attribute_index]
    output = outputs[output_index]
    logit_predictions = (
        output[:, output_prediction_index]
        if attribute_size == 2
        else output[:, :, output_prediction_index]
    )
    probability_predictions = (
        stack([1 - (p := sigmoid(logit_predictions)), p], dim=-1)
        if logit_predictions.dim() == 1
        else softmax(logit_predictions, dim=-1)
    )
    return probability_predictions


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


# def criterion(self, outputs, class_index_targets):
#     loss_terms = defaultdict(float)
#     for class_score_predictions in outputs:
#         if class_score_predictions.dim() == 2:
#             class_count = 2
#             loss_terms["binary"] += binary_cross_entropy_with_logits(
#                 class_score_predictions,
#                 class_index_targets[
#                     :, self.attribute_sizes == class_count
#                 ].float(),
#                 reduction="sum",
#             )
#         else:
#             class_count = class_score_predictions.shape[1]
#             loss_terms[f"categorical_{class_count}"] += cross_entropy(
#                 class_score_predictions,
#                 class_index_targets[:, self.attribute_sizes == class_count],
#                 reduction="sum",
#             )
#     for loss_part_name in loss_terms:
#         loss_terms[loss_part_name] /= (
#             class_index_targets.shape[0] * class_index_targets.shape[1]
#         )
#
#     loss_terms["l2_penalty"] = self.criterion_l2_weight * sum(
#         [parameter.pow(2).sum() for parameter in self.weight_regularisation_parameters]
#     )
#     loss = sum(loss_terms.values())
#     return loss, loss_terms


def train_classifier(
    model_parallel: torch.nn.DataParallel,
    optimizer: torch.optim.Optimizer,
    epoch_count: int,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    experiment: comet_ml.Experiment,
):
    best_model_state = {}
    model = model_parallel.module
    best_valid_loss = None
    device = get_device()
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_eval_state = None
        with experiment.context_manager("train"):
            for data in train_dataloader:
                images, class_index_targets = data[0].to(device), data[1].to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = model_parallel(images)
                class_index_predictions = model.predict(outputs)

                loss, loss_terms = model.criterion(outputs, class_index_targets)
                loss.backward()

                optimizer.step()

                train_eval_results, train_eval_state = evaluate(
                    class_index_predictions,
                    class_index_targets,
                    loss,
                    loss_terms,
                    train_eval_state,
                )
            experiment.log_metrics(train_eval_results, epoch=epoch)

        model_parallel.eval()
        valid_eval_state = None
        with experiment.context_manager("valid"):
            for data in valid_dataloader:
                images, class_index_targets = data[0].to(device), data[1].to(device)

                outputs = model_parallel(images)
                class_index_predictions = model.predict(outputs)

                loss, loss_terms = model.criterion(outputs, class_index_targets)

                valid_eval_results, valid_eval_state = evaluate(
                    class_index_predictions,
                    class_index_targets,
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
