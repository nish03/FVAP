import logging

from piq import FSIMLoss
from torch import float32, int64, no_grad, tensor, zeros

import hpo.Util


def hpo_cost(model, dataloader, sensitive_attribute, alpha=0.5, window_sigma=0.5):
    device = next(model.parameters()).device
    model.eval()

    with no_grad():
        data_index = 0
        dataset_size = len(dataloader.dataset)
        performance_costs = zeros(dataset_size, dtype=float32, device=device)
        labels = zeros(dataset_size, dtype=int64, device=device)
        performance_loss = FSIMLoss(data_range=1.0, chromatic=True, reduction="none")
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            batch_size = len(data)
            data_range = range(data_index, data_index + batch_size)
            data_index += batch_size

            performance_costs[data_range] = performance_loss(output, data)
            labels[data_range] = target

        assert data_index == dataset_size
        entropy_bin_count = max(
            1, int((tensor(performance_costs.shape[0]).log2() - 1).ceil().item())
        )
        entropy_bin_start = performance_costs.min().item()
        entropy_bin_end = performance_costs.max().item()
        performance_entropy = hpo.Util.entropy(
            performance_costs, entropy_bin_count, entropy_bin_start, entropy_bin_end
        )
        mi_performance_sensitive_attribute = performance_entropy.clone()
        sensitive_attribute_entropy = tensor(0.0, device=device)
        for sensitive_attribute_member in sensitive_attribute:
            in_member = labels == sensitive_attribute_member
            probability = in_member.sum() / dataset_size
            mi_performance_sensitive_attribute -= probability * hpo.Util.entropy(
                performance_costs[in_member],
                entropy_bin_count,
                entropy_bin_start,
                entropy_bin_end,
            )
            sensitive_attribute_entropy -= probability * probability.log()
        performance_cost = performance_costs.mean().item()
        fairness_cost = (
            mi_performance_sensitive_attribute
            / (performance_entropy * sensitive_attribute_entropy).sqrt()
        ).item()
        total_cost = (1.0 - alpha) * performance_cost + alpha * fairness_cost

    additional_info = {
        "Performance Cost": performance_cost,
        "Fairness Cost": fairness_cost,
        "Alpha": alpha,
        "Window Sigma": window_sigma,
    }

    logging.debug(
        f"Total Cost: {total_cost} "
        f"Performance Cost: {performance_cost} "
        f"Fairness Cost: {fairness_cost}"
    )
    return total_cost, additional_info
