from abc import ABC, abstractmethod

import torch
from torch import flatten, tensor, sigmoid, softmax, stack
from torch.nn import Linear, Module
from typing import List


class MultiAttributeClassifier(ABC, Module):
    def __init__(self, attribute_sizes: List[int], multi_output_in_filter_count: int):
        Module.__init__(self)
        self.attribute_sizes = tensor(attribute_sizes)
        (
            unique_attribute_sizes,
            inverse_attribute_size_indices,
            attribute_size_frequencies,
        ) = self.attribute_sizes.unique(
            sorted=True, return_inverse=True, return_counts=True
        )
        self.unique_attribute_sizes = unique_attribute_sizes
        self.inverse_attribute_size_indices = inverse_attribute_size_indices
        self.attribute_size_frequencies = attribute_size_frequencies

        multi_output_layer = Module()
        for attribute_size, attribute_size_frequency in zip(
            self.unique_attribute_sizes, self.attribute_size_frequencies
        ):
            multi_output_out_filter_count = attribute_size_frequency.item()
            if attribute_size != 2:
                multi_output_out_filter_count *= attribute_size
            multi_output_module = Linear(
                multi_output_in_filter_count, multi_output_out_filter_count
            )
            multi_output_module_name = (
                "binary"
                if attribute_size == 2
                else f"categorical_{attribute_size.item()}"
            )
            multi_output_layer.add_module(
                multi_output_module_name,
                multi_output_module,
            )
        self.add_module(f"multi_output_layer", multi_output_layer)

    @abstractmethod
    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> (List[torch.Tensor], torch.Tensor):
        output = self.final_layer_output(x)
        multi_output_class_logits = []
        multi_output_class_probabilities = []
        multi_output_label_predictions = []
        for attribute_size, attribute_size_frequency, multi_output_module in zip(
            self.unique_attribute_sizes,
            self.attribute_size_frequencies,
            self.multi_output_layer.children(),
        ):
            class_logits = multi_output_module(flatten(output, 1))
            if attribute_size == 2:
                class_probabilities = sigmoid(class_logits)
                label_predictions = (class_probabilities > 0.5).long()
            else:
                class_logits = class_logits.reshape(
                    -1, attribute_size, attribute_size_frequency
                )
                class_probabilities = softmax(class_logits, dim=1)
                label_predictions = class_probabilities.argmax(dim=1)
            multi_output_class_logits.append(class_logits)
            multi_output_class_probabilities.append(class_probabilities)
            multi_output_label_predictions.append(label_predictions)
        attribute_predictions = []
        for attribute_index, output_index in enumerate(
            self.inverse_attribute_size_indices
        ):
            attribute_size = self.attribute_sizes[attribute_index]
            previous_attribute_sizes = self.attribute_sizes[0:attribute_index]
            prediction_index = previous_attribute_sizes.eq(attribute_size).sum()
            attribute_predictions.append(
                multi_output_label_predictions[output_index][:, prediction_index]
            )
        attribute_predictions = stack(attribute_predictions, dim=1)
        return multi_output_class_logits, multi_output_class_probabilities, attribute_predictions
