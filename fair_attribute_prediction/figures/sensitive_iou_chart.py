#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import tensor

if __name__ == "__main__":
    figure = plt.figure(figsize=(15, 5))
    grid_spec = GridSpec(1, 2)
    color_map = plt.cm.get_cmap("Dark2")

    experiment_types = [
        ("celeba-sensitive_male", "CelebA - Sensitive Male"),
        ("celeba-sensitive_young", "CelebA - Sensitive Young"),
    ]

    loss_types = [
        "$L_{ce}$",
        "$L_{ce} + \\lambda * L_{iou}$",
        "$L_{ce} + \\lambda * L^{l_2}_{eo}$",
        "$L_{ce} + \\lambda * L^{mi}_{eo}$",
        "$L_{ce} + \\lambda * L^{l_2}_{dp}$",
        "$L_{ce} + \\lambda * L^{mi}_{dp}$",
    ]

    loss_type_names = [
        "$L_{ce}$",
        "$L_{iou}$",
        "$L^{l_2}_{eo}$",
        "$L^{mi}_{eo}$",
        "$L^{l_2}_{dp}$",
        "$L^{mi}_{dp}$",
    ]

    figures_data_dir = Path("figures") / "data"
    figures_data_dir.mkdir(parents=True, exist_ok=True)

    for i, (experiment_type_name, title) in enumerate(experiment_types):
        row = i // 2
        column = i % 2
        experiment_descriptions_file_path = figures_data_dir / (experiment_type_name + "-models-processed.json")
        with open(experiment_descriptions_file_path, "r") as experiment_descriptions_file:
            experiment_descriptions = json.load(experiment_descriptions_file)

        loss_types_to_experiment_ids = {}
        for experiment_id, experiment_data in experiment_descriptions["experiments"].items():
            if experiment_id.endswith("_uncalibrated"):
                continue
            loss_types_to_experiment_ids[experiment_data["loss"]] = experiment_id

        stddevs = [
            experiment_descriptions["experiments"][loss_types_to_experiment_ids[loss_type]]["attribute_iou_std"]
            for loss_type in loss_types
        ]

        bar_chart = figure.add_subplot(grid_spec[row, column])
        plt.grid(zorder=0)
        bar_chart.barh(range(len(stddevs)), stddevs, color=color_map.colors, zorder=3)

        bar_chart.set_yticks(range(len(loss_type_names)), loss_type_names if column == 0 else "", fontsize=16)
        bar_chart.set_title(label=title, fontsize=20)
        bar_chart.set_xlabel("Sensitive $IOU_{\\theta}(b)$ Standard Deviation", fontsize=16, y=0.05)
        plt.gca().invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(figures_data_dir / "sensitive_iou_chart.png")
    plt.savefig(figures_data_dir / "sensitive_iou_chart.pdf")
    plt.show()
