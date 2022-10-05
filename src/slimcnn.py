from typing import List

import torch
from torch import cat, tensor, Tensor
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
# see https://arxiv.org/abs/1907.02157 for more details


def depthwise_separable_3x3_conv(input_channel_count: int, output_channel_count: int) -> Sequential:
    """
    Creates a depthwise separable 3x3 convolution.

    :param input_channel_count: Number of input channels
    :param output_channel_count: Number of output channels
    :return: Sequential module that performs the convolution
    """
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
    """
    Creates a new convolution.

    :param input_channel_count: Number of input channels
    :param output_channel_count: Number of output channels
    :param kernel_size: Kernel size
    :param stride: Stride
    :param bias: Bias
    :return: Sequential module that performs the convolution
    """
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
    """
    Perform Xavier uniform initialisation for the weights of Linear and Conv2d modules.
    :param module: Module
    """
    if type(module) in [Linear, Conv2d]:
        xavier_uniform_(module.weight)
        if type(module) == Linear:
            module.bias.data.fill_(0.01)


class SSEBlock(Module):
    """
    Separable Squeeze Expand Block consisting of two 1x1 pointwise convolutional layers and a 3x3 depthwise separable
    convolutional layer.
    """
    def __init__(self, input_channel_count: int, squeeze_filter_count: int, init_weights: bool = False):
        """
        Creates a new SSEBlock instance.

        :param input_channel_count: Number of input channels
        :param squeeze_filter_count: Number of squeeze filters
        :param init_weights: Indicates if the weights should be initialised
        """
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
            depthwise_separable_3x3_conv(self.squeeze_filter_count, self.expand_filter_count),
        )

        if init_weights:
            self.apply(init_module_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass.

        :param x: Tensor containing input features
        :return: Tensor containing output features
        """
        x = self.layer_1_squeeze_conv(x)
        x = cat([self.layer_2_expand_1x1_conv(x), self.layer_2_expand_dw_sep_3x3_conv(x)], 1)
        return x


class SlimModule(Module):
    """
    Slim Module consisting of two SSE blocks a depthwise separable 3x3 convolutional layer and a skip-connection.
    """
    def __init__(self, input_channel_count: int, squeeze_filter_count: int, init_weights: bool = False):
        """
        Creates a new SlimModule instance.

        :param input_channel_count: Number of input channels
        :param squeeze_filter_count: Number of squeeze filters
        :param init_weights: Indicates if the weights should be initialised
        """
        super().__init__()

        expand_filter_count = 4 * squeeze_filter_count
        dw_conv_filters = 3 * squeeze_filter_count

        self.add_module("layer_1_sse_block", SSEBlock(input_channel_count, squeeze_filter_count))
        self.add_module("layer_2_sse_block", SSEBlock(2 * expand_filter_count, squeeze_filter_count))
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
        """
        Performs a forward pass.

        :param x: Tensor containing input features
        :return: Tensor containing output features
        """
        layer_0_x = x
        x = self.layer_1_sse_block(x)
        x = self.layer_2_sse_block(x + self.layer_0_to_1_skip_connection(layer_0_x))
        x = self.layer_3_dw_sep_3x3_conv(x)
        return x


# based on https://github.com/gtamba/pytorch-slim-cnn/blob/master/slimnet.py


class SlimCNN(MultiAttributeClassifier):
    """
    SlimCNN is a computationally-efficient CNN for multi attribute prediction.

    See https://arxiv.org/abs/1907.02157 for more information on the original architecture.
    """
    def __init__(
        self,
        attribute_sizes: List[int],
        attribute_class_weights: List[Tensor],
        squeeze_filter_counts: List[int] = None,
    ):
        """
        Creates a new SlimCNN instance.
        :param attribute_sizes: List[int] contains the class counts for each predicted attribute
        :param attribute_class_weights: List[Tensor] contains a Tensor[attribute_class_count] for each predicted
            attribute that stores the class weighting coefficients (in the range [0, 1] with a sum of 1)
        :param squeeze_filter_counts: (Optional) List[int] Number of squeeze filters for each of the four SlimModules
        """
        if squeeze_filter_counts is None:
            squeeze_filter_counts = [16, 32, 48, 64]
        self.squeeze_filter_counts = tensor(squeeze_filter_counts)

        multi_output_in_filter_count = self.squeeze_filter_counts[-1] * 3
        MultiAttributeClassifier.__init__(self, attribute_sizes, attribute_class_weights, multi_output_in_filter_count)

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
        """
        Adds a convolutional layer to the network.
        """
        self.layer_count += 1
        self.add_module(
            f"layer_{self.layer_count}_convolution",
            conv(3, 96, kernel_size=7, stride=2),
        )

    def add_max_pooling_layer(self):
        """
        Adds a max pooling layer to the network.
        """
        self.layer_count += 1
        self.add_module(f"layer_{self.layer_count}_max_pooling", MaxPool2d(3, stride=2))

    def add_slim_module_layer(self):
        """
        Adds a slim module layer to the network.
        """
        self.layer_count += 1
        self.slim_module_count += 1

        self.add_module(
            f"layer_{self.layer_count}_slim_module",
            SlimModule(
                self.squeeze_filter_counts[self.slim_module_count - 2] * 3 if self.slim_module_count > 1 else 96,
                self.squeeze_filter_counts[self.slim_module_count - 1],
            ),
        )

    def add_global_pooling_layer(self):
        """
        Adds a global pooling layer to the network.
        """
        self.layer_count += 1
        self.add_module(f"layer_{self.layer_count}_global_pooling", AdaptiveAvgPool2d(1))

    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feeds an input Tensor through the network.

        :param x: Input Tensor[N, 3, image_height, image_width] containing N images of arbitrary size
        :return: Tensor[N, base_out_filter_count] containing N feature vectors with as many elements as three times the
            number of squeeze filters in the last SlimModule
        """
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
