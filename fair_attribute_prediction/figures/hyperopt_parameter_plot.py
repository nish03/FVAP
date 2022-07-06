#!/usr/bin/env python3

import json

from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch import tensor

parser = ArgumentParser(fromfile_prefix_chars="+")
parser.add_argument("--experiment_descriptions")
parser.add_argument("--hpo_type")
parser.add_argument("--fig_file_name")


if __name__ == "__main__":
    arguments = parser.parse_args()
    with open(arguments.experiment_descriptions, "r") as experiment_descriptions_file:
        experiment_descriptions = json.load(experiment_descriptions_file)

    baseline_experiment_id = experiment_descriptions["baseline_experiment"]
    reference_experiment_ids = [
        experiment_id
        for experiment_id in experiment_descriptions["experiments"]
        if experiment_id != baseline_experiment_id
    ]

    baseline_attribute_iou_std = experiment_descriptions["experiments"][baseline_experiment_id]["attribute_iou_std"]
    baseline_accuracy = experiment_descriptions["experiments"][baseline_experiment_id]["accuracy"]
    reference_fair_loss_weights = []
    reference_attribute_iou_stds = []
    reference_accuracies = []

    for reference_experiment_id in sorted(reference_experiment_ids):
        reference_fair_loss_weights.append(
            experiment_descriptions["experiments"][reference_experiment_id]["fair_loss_weight"]
        )
        reference_attribute_iou_stds.append(
            experiment_descriptions["experiments"][reference_experiment_id]["attribute_iou_std"]
        )
        reference_accuracies.append(experiment_descriptions["experiments"][reference_experiment_id]["accuracy"])

    best_reference_experiment_index = tensor(reference_attribute_iou_stds).argmin()
    best_reference_experiment_id = list(sorted(reference_experiment_ids))[best_reference_experiment_index]
    best_reference_fair_loss_weight = reference_fair_loss_weights[best_reference_experiment_index]
    best_reference_attribute_iou_std = reference_attribute_iou_stds[best_reference_experiment_index]
    best_reference_accuracy = reference_accuracies[best_reference_experiment_index]
    print(
        f"λ={0.0:.3E}, IoU-Std={baseline_attribute_iou_std:.3E}, Accuracy={baseline_accuracy:.3E} ({baseline_experiment_id})"
    )
    print(
        f"λ={best_reference_fair_loss_weight:.3E}, IoU-Std={best_reference_attribute_iou_std:.3E}, Accuracy={best_reference_accuracy:.3E} ({best_reference_experiment_id})"
    )

    figures_data_dir = Path("figures") / "data"
    figures_data_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.scatter(reference_fair_loss_weights, reference_attribute_iou_stds, c=reference_accuracies, cmap="viridis")
    plt.axhline(baseline_attribute_iou_std, label="Unfair Baseline")
    plt.xscale("log")
    plt.xlabel("Fair Loss Weighting Coefficient $\\lambda$", fontsize=16)
    plt.ylabel("Sensitive $IOU_{\\theta}(b)$ Standard Deviation", fontsize=16)
    plt.suptitle(arguments.hpo_type, fontsize=20)
    plt.grid()
    plt.legend()
    plt.tight_layout(rect=[0,0, 0.95, 1])
    cb = plt.colorbar(format=PercentFormatter(xmax=1.0), orientation="vertical")
    cb.set_label(label="Accuracy", fontsize=16)
    cb.ax.plot([0, 1], [baseline_accuracy] * 2, label="Unfair Baseline")
    plt.savefig(figures_data_dir / f"{arguments.fig_file_name}_hpo.png")
    plt.savefig(figures_data_dir / f"{arguments.fig_file_name}_hpo.pdf")
    plt.show()
