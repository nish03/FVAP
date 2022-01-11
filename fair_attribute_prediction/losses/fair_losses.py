from losses.fair_equality_of_opportunity_loss import fair_equality_of_opportunity_loss
from losses.fair_equalized_odds_loss import fair_equalized_odds_loss
from losses.fair_intersection_over_union import fair_intersection_over_union_loss
from losses.fair_mutual_information_loss import fair_mutual_information_3_loss, fair_mutual_information_loss

fair_losses = {
    "equality_of_opportunity": fair_equality_of_opportunity_loss,
    "equalized_odds": fair_equalized_odds_loss,
    "intersection_over_union": fair_intersection_over_union_loss,
    "mutual_information": fair_mutual_information_loss,
    "mutual_information_3": fair_mutual_information_3_loss,
}
