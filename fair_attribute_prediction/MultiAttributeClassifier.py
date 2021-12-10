from abc import ABC, abstractmethod
from torch.nn import Module, Linear
from torch import tensor, flatten, stack

import torch


class MultiAttributeClassifier(ABC, Module):
    def __init__(self, attribute_sizes: list[int]):
        Module.__init__(self)
        self.attribute_sizes = tensor(attribute_sizes)
        (
            self.unique_attribute_sizes,
            self.inverse_attribute_size_indices,
            self.attribute_size_counts,
        ) = self.attribute_sizes.unique(
            sorted=True, return_inverse=True, return_counts=True
        )

    def add_output_layer(self, in_filter_count):
        output_layer = Module()
        for attribute_size, attribute_size_count in zip(
            self.unique_attribute_sizes, self.attribute_size_counts
        ):
            out_filter_count = attribute_size_count
            if attribute_size != 2:
                out_filter_count *= attribute_size
            output_module = Linear(in_filter_count, out_filter_count)
            output_module_name = (
                "binary"
                if attribute_size == 2
                else f"categorical_{attribute_size.item()}"
            )
            output_layer.add_module(
                output_module_name,
                output_module,
            )
        self.add_module(f"output_layer", output_layer)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError()

    def create_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for attribute_size, attribute_size_count, output_module in zip(
            self.unique_attribute_sizes,
            self.attribute_size_counts,
            self.output_layer.children(),
        ):
            output = output_module(flatten(x, 1))
            if attribute_size != 2:
                output.reshape(-1, attribute_size, attribute_size_count)
            outputs.append(output)
        return outputs
    
    def output_prediction_index(self, attribute_index: int) -> int:

    def predict(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        outputs_class_predictions = []
        for output_logit_predictions in outputs:
            if output_logit_predictions.dim() == 2:
                output_class_predictions = tensor(
                    output_logit_predictions > 0.0, dtype=torch.long
                )
            else:
                output_class_predictions = output_logit_predictions.argmax(dim=1)
            outputs_class_predictions.append(output_class_predictions)
        class_predictions = []
        for attribute_index, output_index in enumerate(
            self.inverse_attribute_size_indices
        ):
            attribute_size = self.attribute_sizes[attribute_index]
            previous_attribute_sizes = self.attribute_sizes[0:attribute_index]
            output_prediction_index = tensor(
                previous_attribute_sizes == attribute_size,
                dtype=torch.int64,
            ).sum()
            class_predictions.append(
                outputs_class_predictions[output_index][:, output_prediction_index]
            )
        class_predictions = stack(class_predictions, dim=1)
        return class_predictions
