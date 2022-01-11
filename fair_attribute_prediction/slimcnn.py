from collections import defaultdict

import torch
from torch import cat, flatten, stack, tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.init import xavier_uniform_

from multi_attribute_classifier import MultiAttributeClassifier

# based on https://github.com/gtamba/pytorch-slim-cnn/blob/master/layers.py


def depthwise_separable_3x3_conv(input_channel_count, output_channel_count):
    return Sequential(
        Conv2d(
            input_channel_count,
            input_channel_count,
            kernel_size=(3, 3),
            padding=1,
            groups=input_channel_count,
            bias=False,
        ),
        BatchNorm2d(input_channel_count),
        ReLU(inplace=True),
        Conv2d(
            input_channel_count,
            output_channel_count,
            kernel_size=(1, 1),
            bias=False,
        ),
        BatchNorm2d(output_channel_count),
        ReLU(inplace=True),
    )


def conv(input_channel_count, output_channel_count, kernel_size, stride=1, bias=True):
    return Sequential(
        Conv2d(
            input_channel_count,
            output_channel_count,
            kernel_size=kernel_size,
            stride=(stride, stride),
            bias=bias,
        ),
        BatchNorm2d(output_channel_count),
        ReLU(inplace=True),
    )


def init_module_weights(module):
    if type(module) in [Linear, Conv2d]:
        xavier_uniform_(module.weight)
        if type(module) == Linear:
            module.bias.data.fill_(0.01)


class SSEBlock(Module):
    def __init__(self, input_channel_count, squeeze_filter_count, init_weights=False):
        super().__init__()

        self.input_channel_count = input_channel_count
        self.squeeze_filter_count = squeeze_filter_count
        self.expand_filter_count = 4 * squeeze_filter_count
        self.output_channel_count = 2 * self.expand_filter_count

        self.add_module(
            "layer_1_squeeze_conv",
            conv(self.input_channel_count, self.squeeze_filter_count, kernel_size=1),
        )
        self.add_module(
            "layer_2_expand_1x1_conv",
            conv(self.squeeze_filter_count, self.expand_filter_count, kernel_size=1),
        )
        self.add_module(
            "layer_2_expand_dw_sep_3x3_conv",
            depthwise_separable_3x3_conv(
                self.squeeze_filter_count, self.expand_filter_count
            ),
        )

        if init_weights:
            self.apply(init_module_weights)

    def forward(self, x):
        x = self.layer_1_squeeze_conv(x)
        x = cat(
            [self.layer_2_expand_1x1_conv(x), self.layer_2_expand_dw_sep_3x3_conv(x)], 1
        )
        return x


class SlimModule(Module):
    def __init__(self, input_channel_count, squeeze_filter_count, init_weights=False):
        super().__init__()

        expand_filter_count = 4 * squeeze_filter_count
        dw_conv_filters = 3 * squeeze_filter_count

        self.add_module(
            "layer_1_sse_block", SSEBlock(input_channel_count, squeeze_filter_count)
        )
        self.add_module(
            "layer_2_sse_block", SSEBlock(2 * expand_filter_count, squeeze_filter_count)
        )
        self.add_module(
            "layer_3_dw_sep_3x3_conv",
            depthwise_separable_3x3_conv(2 * expand_filter_count, dw_conv_filters),
        )

        self.add_module(
            "layer_0_to_1_skip_connection",
            conv(
                input_channel_count,
                2 * expand_filter_count,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

        if init_weights:
            self.apply(init_module_weights)

    def forward(self, x):
        layer_0_x = x
        x = self.layer_1_sse_block(x)
        x = self.layer_2_sse_block(x + self.layer_0_to_1_skip_connection(layer_0_x))
        x = self.layer_3_dw_sep_3x3_conv(x)
        return x


# based on https://github.com/gtamba/pytorch-slim-cnn/blob/master/slimnet.py


class SlimCNN(MultiAttributeClassifier):
    def __init__(
        self,
        squeeze_filter_counts=None,
        attribute_sizes=None,
    ):

        if squeeze_filter_counts is None:
            squeeze_filter_counts = [16, 32, 48, 64]
        self.squeeze_filter_counts = tensor(squeeze_filter_counts)

        multi_output_in_filter_count = (
                self.squeeze_filter_counts[- 1] * 3
        )
        MultiAttributeClassifier.__init__(
            self, attribute_sizes, multi_output_in_filter_count
        )

        self.layer_count = 0
        self.slim_module_count = 0
        self.add_convolution_layer()
        self.add_max_pooling_layer()
        self.add_slim_module_layer()
        self.add_max_pooling_layer()
        self.add_slim_module_layer()
        self.add_max_pooling_layer()
        self.add_slim_module_layer()
        self.add_max_pooling_layer()
        self.add_slim_module_layer()
        self.add_max_pooling_layer()
        self.add_global_pooling_layer()

        self.apply(init_module_weights)

    def add_convolution_layer(self):
        self.layer_count += 1
        self.add_module(
            f"layer_{self.layer_count}_convolution",
            conv(3, 96, kernel_size=7, stride=2),
        )

    def add_max_pooling_layer(self):
        self.layer_count += 1
        self.add_module(f"layer_{self.layer_count}_max_pooling", MaxPool2d(3, stride=2))

    def add_slim_module_layer(self):
        self.layer_count += 1
        self.slim_module_count += 1

        self.add_module(
            f"layer_{self.layer_count}_slim_module",
            SlimModule(
                self.squeeze_filter_counts[self.slim_module_count - 2] * 3
                if self.slim_module_count > 1
                else 96,
                self.squeeze_filter_counts[self.slim_module_count - 1],
            ),
        )

    def add_global_pooling_layer(self):
        self.layer_count += 1
        self.add_module(
            f"layer_{self.layer_count}_global_pooling", AdaptiveAvgPool2d(1)
        )

    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layer_1_convolution(x)
        output = self.layer_2_max_pooling(output)
        output = self.layer_3_slim_module(output)
        output = self.layer_4_max_pooling(output)
        output = self.layer_5_slim_module(output)
        output = self.layer_6_max_pooling(output)
        output = self.layer_7_slim_module(output)
        output = self.layer_8_max_pooling(output)
        output = self.layer_9_slim_module(output)
        output = self.layer_10_max_pooling(output)
        output = self.layer_11_global_pooling(output)
        return output
