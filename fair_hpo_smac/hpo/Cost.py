import logging

from torch import float32, int64, no_grad, zeros, tensor
from numpy import histogram_bin_edges

from model.util.ReconstructionLoss import MultiScaleSSIMLoss


def entropy(samples):
    bin_count = len(histogram_bin_edges(samples.cpu(), bins="auto")) - 1
    probabilities = samples.histc(bins=bin_count)
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities.nonzero(as_tuple=True)]
    return -(probabilities * probabilities.log()).sum()


def hpo_cost(model, dataloader, sensitive_attribute, alpha=0.5, window_sigma=0.5):
    device = next(model.parameters()).device
    model.eval()

    with no_grad():
        data_index = 0
        dataset_size = len(dataloader.dataset)
        ms_ssim_scores = zeros(dataset_size, dtype=float32, device=device)
        labels = zeros(dataset_size, dtype=int64, device=device)
        ms_ssim_loss = MultiScaleSSIMLoss(window_sigma=window_sigma, reduction="none")
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            batch_size = len(data)
            data_range = range(data_index, data_index + batch_size)
            data_index += batch_size

            ms_ssim_scores[data_range] = 1.0 - ms_ssim_loss(output, data)
            labels[data_range] = target

        assert data_index == dataset_size

        ms_ssim_entropy = entropy(ms_ssim_scores)
        mi_ms_ssim_sensitive_attribute = ms_ssim_entropy.clone()
        sensitive_attribute_entropy = tensor(0.0, device=device)
        for sensitive_attribute_member in sensitive_attribute:
            in_member = labels == sensitive_attribute_member
            probability = in_member.sum() / dataset_size
            mi_ms_ssim_sensitive_attribute -= probability * entropy(
                ms_ssim_scores[in_member]
            )
            sensitive_attribute_entropy -= probability * probability.log()
        performance_cost = 1.0 - ms_ssim_scores.mean().item()
        fairness_cost = (
            mi_ms_ssim_sensitive_attribute
            / (ms_ssim_entropy + sensitive_attribute_entropy).sqrt()
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
