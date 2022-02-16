from losses.fair_demographic_parity_loss import fair_demographic_parity_loss
from losses.fair_equalized_odds_loss import fair_equalized_odds_loss
from losses.fair_intersection_over_union import fair_intersection_over_union_loss
from losses.fair_mutual_information_dp_loss import fair_mutual_information_dp_loss

fair_losses = {
    "demographic_parity": fair_demographic_parity_loss,
    "equalized_odds": fair_equalized_odds_loss,
    "intersection_over_union": fair_intersection_over_union_loss,
    "mutual_information_dp": fair_mutual_information_dp_loss,
}
