from typing import Dict, List, Optional, Tuple
import torch

from Metrics import MetricsState, metrics
from MultiAttributeDataset import Attribute
from Util import get_device
from losses.CrossEntropyLoss import cross_entropy_loss
from losses.FairEqualityOfOpportunityLoss import fair_equality_of_opportunity_loss
from losses.FairEqualizedOddsLoss import fair_equalized_odds_loss
from losses.FairIntersectionOverUnionLoss import fair_intersection_over_union_loss
from losses.FairMutualInformationLoss import fair_mutual_information_3_loss, fair_mutual_information_loss

fair_losses = {
    "equality_of_opportunity": fair_equality_of_opportunity_loss,
    "equalized_odds": fair_equalized_odds_loss,
    "intersection_over_union": fair_intersection_over_union_loss,
    "mutual_information": fair_mutual_information_loss,
    "mutual_information_3": fair_mutual_information_3_loss,
}


def loss(
    model: torch.nn.Module,
    multi_output_class_logits: List[torch.Tensor],
    multi_attribute_targets: torch.Tensor,
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    fair_loss_type: str,
    fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _cross_entropy_loss = cross_entropy_loss(model, multi_output_class_logits, multi_attribute_targets)

    fair_loss = fair_losses[fair_loss_type]
    _fair_loss = fair_loss(
        model, multi_output_class_logits, multi_attribute_targets, sensitive_attribute, target_attribute
    )

    _loss = _cross_entropy_loss + fair_loss_weight * _fair_loss
    loss_term_values = {"cross_entropy": _cross_entropy_loss.item(), f"fair_{fair_loss_type}": _fair_loss.item()}

    return _loss, loss_term_values


def loss_with_metrics(
    model: torch.nn.Module,
    batch_data: (torch.Tensor, torch.Tensor),
    metrics_state: Optional[MetricsState],
    sensitive_attribute: Attribute,
    target_attribute: Attribute,
    fair_loss_type: str,
    fair_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], MetricsState]:
    images, multi_attribute_targets = batch_data[0].to(get_device()), batch_data[1].to(get_device())

    multi_output_class_logits = model(images)

    _loss, loss_term_values = loss(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        sensitive_attribute,
        target_attribute,
        fair_loss_type,
        fair_loss_weight,
    )

    loss_value = _loss.item()
    _metrics, metrics_state = metrics(
        model,
        multi_output_class_logits,
        multi_attribute_targets,
        loss_value,
        loss_term_values,
        metrics_state,
    )
    return _loss, _metrics, metrics_state
