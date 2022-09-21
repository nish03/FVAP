from abc import ABC, abstractmethod

import torch
from torch import flatten, tensor, sigmoid, softmax, stack
from torch.nn import Linear, Module
from typing import List, Tuple


class MultiAttributeClassifier(ABC, Module):
    def __init__(
        self, attribute_sizes: List[int], attribute_class_weights: List[List[int]], multi_output_in_filter_count: int
    ):
        Module.__init__(self)
        self.attribute_sizes = tensor(attribute_sizes)
        self.attribute_class_weights = attribute_class_weights
        (
            unique_attribute_sizes,
            inverse_attribute_size_indices,
            attribute_size_frequencies,
        ) = self.attribute_sizes.unique(sorted=True, return_inverse=True, return_counts=True)
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
            multi_output_module = Linear(multi_output_in_filter_count, multi_output_out_filter_count)
            multi_output_module_name = "binary" if attribute_size == 2 else f"categorical_{attribute_size.item()}"
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
        for attribute_size, attribute_size_frequency, multi_output_module in zip(
            self.unique_attribute_sizes, self.attribute_size_frequencies, self.multi_output_layer.children()
        ):
            output_class_logits = multi_output_module(flatten(output, 1))
            if attribute_size != 2:
                output_class_logits = output_class_logits.reshape(-1, attribute_size, attribute_size_frequency)
            multi_output_class_logits.append(output_class_logits)
        return multi_output_class_logits

    def multi_output_indices(self, attribute_index: int) -> Tuple[int, int]:
        multi_output_index = self.inverse_attribute_size_indices[attribute_index].item()
        attribute_size = self.attribute_sizes[attribute_index]
        previous_attribute_sizes = self.attribute_sizes[0:attribute_index]
        output_index = previous_attribute_sizes.eq(attribute_size).sum().item()
        return multi_output_index, output_index

    def multi_attribute_predictions(self, multi_output_class_logits: List[torch.Tensor]) -> torch.Tensor:
        multi_output_attribute_predictions = []
        for output_class_logits in multi_output_class_logits:
            if output_class_logits.dim() == 2:
                attribute_predictions = (output_class_logits > 0.0).long()
            else:
                attribute_predictions = output_class_logits.argmax(dim=1)
            multi_output_attribute_predictions.append(attribute_predictions)
        _multi_attribute_predictions = []
        # reorder attribute predictions from the model outputs into their original order
        for attribute_index in range(len(self.attribute_sizes)):
            multi_output_index, output_index = self.multi_output_indices(attribute_index)
            attribute_predictions = multi_output_attribute_predictions[multi_output_index][:, output_index]
            _multi_attribute_predictions.append(attribute_predictions)
        _multi_attribute_predictions = stack(_multi_attribute_predictions, dim=-1)
        return _multi_attribute_predictions

    def attribute_class_probabilities(
        self, multi_output_class_logits: List[torch.Tensor], attribute_index: int
    ) -> torch.Tensor:
        multi_output_index, output_index = self.multi_output_indices(attribute_index)
        class_logits = multi_output_class_logits[multi_output_index][..., output_index]
        if class_logits.dim() == 1:
            class_1_probabilities = sigmoid(class_logits)
            class_0_probabilities = 1 - class_1_probabilities
            _attribute_probabilities = stack([class_0_probabilities, class_1_probabilities], dim=1)
        else:
            _attribute_probabilities = softmax(class_logits, dim=1)
        return _attribute_probabilities
