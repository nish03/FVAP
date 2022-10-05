from losses.fair_demographic_parity_loss import fair_demographic_parity_loss
from losses.fair_equalized_odds_loss import fair_equalized_odds_loss
from losses.fair_intersection_over_union_loss import (
    fair_intersection_over_union_paired_loss,
    fair_intersection_over_union_conditioned_loss,
)
from losses.fair_mutual_information_loss import fair_mutual_information_dp_loss, fair_mutual_information_eo_loss

""" Dict[str,function] maps fair loss names to functions. """
fair_losses = {
    "demographic_parity": fair_demographic_parity_loss,
    "equalized_odds": fair_equalized_odds_loss,
    "intersection_over_union_paired": fair_intersection_over_union_paired_loss,
    "intersection_over_union_conditioned": fair_intersection_over_union_conditioned_loss,
    "mutual_information_dp": fair_mutual_information_dp_loss,
    "mutual_information_eo": fair_mutual_information_eo_loss,
}
