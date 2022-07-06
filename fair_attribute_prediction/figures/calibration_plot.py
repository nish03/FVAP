#!/usr/bin/env python3

import json
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import numpy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

parser = ArgumentParser(fromfile_prefix_chars="+")
parser.add_argument("--experiment_descriptions")

if __name__ == "__main__":
    arguments = parser.parse_args()
    experiment_descriptions_file_path = Path(arguments.experiment_descriptions)
    with open(experiment_descriptions_file_path, "r") as experiment_descriptions_file:
        experiment_descriptions = json.load(experiment_descriptions_file)
    calibration_plot_figure = plt.figure(
        figsize=(12, int(2 + ceil(len(experiment_descriptions["experiments"]) / 2.0)) * 3)
    )
    grid_spec = GridSpec(2 + ceil(len(experiment_descriptions["experiments"]) / 2.0), 2)
    colors = plt.cm.get_cmap("Dark2")

    calibration_curve_axes = calibration_plot_figure.add_subplot(grid_spec[:2, :2])
    calibration_displays = {}
    for i, (experiment_id, experiment_data) in enumerate(sorted(experiment_descriptions["experiments"].items())):
        loss_name = experiment_data["loss"]
        display = CalibrationDisplay.from_predictions(
            numpy.array(experiment_data["target_attribute_values"]),
            numpy.array(experiment_data["target_class_probabilities"])[:, 1],
            n_bins=5,
            name=loss_name,
            ax=calibration_curve_axes,
            color=colors(i),
        )
        calibration_displays[experiment_id] = display

    calibration_curve_axes.grid()
    calibration_curve_axes.set_title(
        f"Calibration plots ({experiment_descriptions['model']} - {experiment_descriptions['dataset']})"
    )

    # Add histogram
    for i, (experiment_id, experiment_data) in enumerate(sorted(experiment_descriptions["experiments"].items())):
        row = 2 + int(i / 2)
        column = i % 2
        histogram_axes = calibration_plot_figure.add_subplot(grid_spec[row, column])

        histogram_axes.hist(
            calibration_displays[experiment_id].y_prob,
            range=(0, 1),
            bins=10,
            label=experiment_data["loss"],
            color=colors(i),
        )
        histogram_axes.set(title=experiment_data["loss"], xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig(
        experiment_descriptions_file_path.parent
        / ("-".join(experiment_descriptions_file_path.stem.split("-")[:-1]) + "-calibration_plot.png")
    )
    plt.show()
