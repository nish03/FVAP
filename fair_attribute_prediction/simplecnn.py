import torch
from torch import relu
from torch.nn import Conv2d, BatchNorm2d, AdaptiveAvgPool2d

from multi_attribute_classifier import MultiAttributeClassifier


class SimpleCNN(MultiAttributeClassifier):
    def __init__(
        self,
        hidden_layer_count=4,
        base_out_filter_count=64,
        attribute_sizes=None,
        attribute_class_weights=None,
    ):

        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_padding = 1
        self.hidden_layer_stride = (2, 2)
        self.kernel_size = (4, 4)
        self.base_out_filter_count = base_out_filter_count

        MultiAttributeClassifier.__init__(
            self,
            attribute_sizes,
            attribute_class_weights,
            multi_output_in_filter_count=self.base_out_filter_count * 2 ** (self.hidden_layer_count - 1),
        )

        for layer_index in range(1, self.hidden_layer_count + 1):
            in_filter_count = self.base_out_filter_count * 2 ** (layer_index - 2) if layer_index != 1 else 3
            out_filter_count = (
                self.base_out_filter_count * 2 ** (layer_index - 1) if layer_index != 1 else self.base_out_filter_count
            )
            convolution_module = Conv2d(
                in_filter_count,
                out_filter_count,
                self.kernel_size,
                stride=self.hidden_layer_stride,
                padding=self.hidden_layer_padding,
            )
            batch_norm_module = BatchNorm2d(out_filter_count)
            self.add_module(f"layer_{layer_index}_convolution", convolution_module)
            self.add_module(f"layer_{layer_index}_batch_norm", batch_norm_module)

        global_pooling_module = AdaptiveAvgPool2d(output_size=1)
        self.add_module(f"layer_{self.hidden_layer_count}_global_pooling", global_pooling_module)

    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        for layer_index in range(1, self.hidden_layer_count + 1):
            convolution_module = self.get_submodule(f"layer_{layer_index}_convolution")
            batch_norm_module = self.get_submodule(f"layer_{layer_index}_batch_norm")
            x = relu(batch_norm_module(convolution_module(x)))
        global_pooling_module = self.get_submodule(f"layer_{self.hidden_layer_count}_global_pooling")
        x = global_pooling_module(x)
        return x
