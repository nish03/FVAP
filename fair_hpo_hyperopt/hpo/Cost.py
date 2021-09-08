import logging
from model.util.ReconstructionLoss import MultiScaleSSIMLoss
from torch import float32, zeros, no_grad, tensor, int64, arange


def ms_ssim_cost(_model, _dataloader, _, window_sigma=0.5):
    device = next(_model.parameters()).device
    _model.eval()

    cost = zeros(1, dtype=float32)
    processed_data_samples = 0
    ms_ssim_loss = MultiScaleSSIMLoss(window_sigma=window_sigma, reduction="sum")
    with no_grad():
        for data, _ in _dataloader:
            data = data.to(device)
            output = _model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            cost += ms_ssim_loss(output, data).item()
            processed_data_samples += len(data)
    cost = (cost / processed_data_samples).item()
    additional_info = {"Window Sigma": window_sigma}
    logging.debug(f"  MS-SSIM Cost: {cost}")
    return cost, additional_info


def fair_ms_ssim_cost(_model, _dataloader, sensitive_attribute, window_sigma=0.5):
    device = next(_model.parameters()).device
    _model.eval()

    costs = zeros(len(sensitive_attribute), dtype=float32)
    processed_data_samples = zeros(len(sensitive_attribute), dtype=int64)
    ms_ssim_loss = MultiScaleSSIMLoss(window_sigma=window_sigma, reduction="sum")
    with no_grad():
        for data, target in _dataloader:
            data, target = data.to(device), target.to(device)
            output = _model.reconstruct(data)
            data = (data + 1.0) / 2.0
            output = (output + 1.0) / 2.0
            costs += tensor(
                [
                    ms_ssim_loss(
                        output[target == member.value],
                        data[target == member.value],
                    )
                    for member in sensitive_attribute
                ]
            )
            processed_data_samples += (
                (target == arange(len(sensitive_attribute), device=device).view(-1, 1))
                .count_nonzero(dim=1)
                .cpu()
            )

    costs /= processed_data_samples
    cost = costs.max().item()
    sensitive_attribute_costs = {
        str(member): costs[member.value].item() for member in sensitive_attribute
    }
    additional_info = {
        "Sensitive Attribute Costs": sensitive_attribute_costs,
        "Window Sigma": window_sigma,
    }
    for member in sensitive_attribute:
        logging.debug(f"  MS-SSIM Cost[{str(member)}]: {costs[member.value].item()}")
    logging.debug(f"  Fair MS-SSIM Cost: {cost}")
    return cost, additional_info


cost_functions = {"MS-SSIM": ms_ssim_cost, "FairMS-SSIM": fair_ms_ssim_cost}
