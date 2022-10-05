from typing import Dict, List, Optional, Tuple
import torch
from torch import no_grad

from losses.cross_entropy_loss import cross_entropy_loss
from losses.fair_losses import fair_losses
from metrics import MetricsState, metrics
from multi_attribute_dataset import Attribute
from util import get_device


def losses(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    attributes: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    prediction_attribute_indices: List[int],
    optimized_fair_loss_type: str,
    optimized_fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    Computes the optimized loss and all fair losses for given samples.

    :param model: Module / MultiAttributeClassifier predicting the logits for each attribute
    :param multi_output_class_logits: List(Tensor[sample_count, attribute_count(, attribute_class_count)])
        containing as many tensors as there are unique attribute sizes.
        All Tensors contain the predicted logits for each sample, attribute of the given size and attribute class.
        The tensor for binary attributes has only 2 dimensions as the logits for the other class can be deduced.
    :param attributes: Tensor[sample_count, overall_attribute_count]
        containing the ground truth labels for each sample and attribute
    :param sensitive_attribute: Sensitive Attribute
    :param target_attribute: Target Attribute
    :param prediction_attribute_indices: Indices of the predicted attributes (into the 2. dim of attributes)
    :param optimized_fair_loss_type: Name of the optimized fair loss, can be one of the following:
         "demographic_parity", "equalized_odds", "intersection_over_union_paired",
         "intersection_over_union_conditioned", "mutual_information_dp", "mutual_information_eo"
    :param optimized_fair_loss_weight: Weight that specifies the "importance" of the fair loss term
    :return: Tensor[] containing the differentiable optimized loss value,
             Dict[str, float] maps each fair loss name to its value,
             Tensor[sample_count, prediction_attribute_count] containing the ground truth labels for each sample and
                prediction attribute
    """
    multi_attribute_targets = attributes[:, prediction_attribute_indices]
    optimized_cross_entropy_loss = cross_entropy_loss(model, multi_output_class_logits, multi_attribute_targets)

    sensitive_attribute.targets = attributes[:, sensitive_attribute.index]
    target_attribute.targets = attributes[:, target_attribute.index]
    target_prediction_attribute_index = prediction_attribute_indices.index(target_attribute.index)
    target_attribute.class_probabilities = model.module.attribute_class_probabilities(
        multi_output_class_logits, target_prediction_attribute_index
    )
    optimized_fair_loss = fair_losses[optimized_fair_loss_type](sensitive_attribute, target_attribute)

    optimized_loss = optimized_cross_entropy_loss + optimized_fair_loss_weight * optimized_fair_loss

    with no_grad():
        additional_losses = {
            "cross_entropy": optimized_cross_entropy_loss.item(),
            f"fair_{optimized_fair_loss_type}": optimized_fair_loss.item(),
        }

        for additional_fair_loss_type in fair_losses:
            if additional_fair_loss_type == optimized_fair_loss_type:
                continue
            additional_fair_loss = fair_losses[additional_fair_loss_type](sensitive_attribute, target_attribute)
            additional_losses[f"fair_{additional_fair_loss_type}"] = additional_fair_loss.item()

    return optimized_loss, additional_losses, multi_attribute_targets


def losses_with_metrics(
    model: torch.nn.Module,
    batch_data: (torch.Tensor, torch.Tensor),
    metrics_state: Optional[MetricsState],
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    prediction_attribute_indices: List[int],
    optimized_fair_loss_type: str,
    optimized_fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], MetricsState]:
    """
    Computes the optimized loss and metrics for a batch of samples (images and ground truth attribute class labels).

    :param model: Module / MultiAttributeClassifier predicting the logits for each attribute
    :param batch_data: (Tensor[sample_count, colour_channel_count, image_width, image_height],
                        Tensor[sample_count, attribute_count])
        containing images and ground truth labels for each batch sample and attribute class
    :param metrics_state: MetricsState storing the state of the metric computation
        This function will return an initialized state if this parameter is set to None.
    :param sensitive_attribute: Sensitive Attribute
    :param target_attribute: Target Attribute
    :param prediction_attribute_indices: Indices of the predicted attributes (into the 2. dim of attributes)
    :param optimized_fair_loss_type: Name of the optimized fair loss, can be one of the following:
         "demographic_parity", "equalized_odds", "intersection_over_union_paired",
         "intersection_over_union_conditioned", "mutual_information_dp", "mutual_information_eo"
    :param optimized_fair_loss_weight: Weight that specifies the "importance" of the fair loss term
    :return: Tensor[] containing the differentiable optimized loss value,
             Dict[str, float] maps each metric name to its value,
             MetricState contains the updated/initialized version of metrics_state
    """
    images, attributes = batch_data[0].to(get_device()), batch_data[1].to(get_device())

    multi_output_class_logits = model(images)

    optimized_loss, additional_losses, multi_attribute_targets = losses(
        model,
        multi_output_class_logits,
        attributes,
        sensitive_attribute,
        target_attribute,
        prediction_attribute_indices,
        optimized_fair_loss_type,
        optimized_fair_loss_weight,
    )

    _metrics, metrics_state = metrics(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        optimized_loss.item(),
        additional_losses,
        metrics_state,
    )
    return optimized_loss, _metrics, metrics_state
