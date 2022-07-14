#!/usr/bin/env python3

import json
from pathlib import Path

from matplotlib import pyplot as plt


if __name__ == "__main__":
    figure = plt.figure(figsize=(12, 8.5))
    color_map = plt.cm.get_cmap("Dark2")

    experiment_types = [
        ("celeba-sensitive_male", "CelebA - Male"),
        ("celeba-sensitive_young", "CelebA - Young"),
    ]

    loss_type_names = [
        "$L_{ce}$",
        "$L_{iou}$",
        "$L^{l_2}_{eo}$",
        "$L^{mi}_{eo}$",
        "$L^{l_2}_{dp}$",
        "$L^{mi}_{dp}$",
    ]

    loss_types = [
        "$L_{ce}$",
        "$L_{ce} + \\lambda * L_{iou}$",
        "$L_{ce} + \\lambda * L^{l_2}_{eo}$",
        "$L_{ce} + \\lambda * L^{mi}_{eo}$",
        "$L_{ce} + \\lambda * L^{l_2}_{dp}$",
        "$L_{ce} + \\lambda * L^{mi}_{dp}$",
    ]

    for i, (experiment_type_name, title) in enumerate(experiment_types):
        experiment_descriptions_file_path = Path("figures") / "data" / (experiment_type_name + "-models-processed.json")
        with open(experiment_descriptions_file_path, "r") as experiment_descriptions_file:
            experiment_descriptions = json.load(experiment_descriptions_file)

        loss_types_to_experiment_ids = {}
        for experiment_id, experiment_data in experiment_descriptions["experiments"].items():
            if experiment_id.endswith("_uncalibrated"):
                continue
            loss_types_to_experiment_ids[experiment_data["loss"]] = experiment_id

        accuracies = [
            # f'{experiment_descriptions["experiments"][loss_types_to_experiment_ids[loss_type]]["accuracy"]:.3}'
            experiment_descriptions["experiments"][loss_types_to_experiment_ids[loss_type]]["accuracy"]
            for loss_type in loss_types
        ]

        x_positions = [accuracy_index + i * 0.4 - 0.2 for accuracy_index in range(len(accuracies))]
        plt.bar(
            x_positions,
            accuracies,
            0.35,
            color=color_map.colors,
            label=title,
        )
        print(f"{accuracies=} {experiment_type_name=}")

    plt.xticks(range(len(loss_type_names)), loss_type_names)
    plt.ylim([0.85, 0.95])

    figures_data_dir = Path("figures") / "data"
    figures_data_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Accuracies")
    plt.savefig(figures_data_dir / "accuracy_chart.png")
    plt.savefig(figures_data_dir / "accuracy_chart.pdf")
    plt.show()
