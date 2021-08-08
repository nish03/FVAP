from numpy import argmin, array
from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, Linear, ReLU
from torch.nn.functional import avg_pool2d


class ResidualUnitBN(Module):
    layer_count = 2

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnitBN, self).__init__()
        self.conv1 = Sequential(
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=1,
                bias=False,
            ),
        )
        self.conv2 = Sequential(
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
        )
        self.shortcut = Sequential()
        if in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    padding=0,
                    bias=False,
                )
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class ResidualUnit(Module):
    layer_count = 2

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnit, self).__init__()
        self.conv1 = Sequential(
            ReLU(inplace=True),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=1,
                bias=False,
            ),
        )
        self.conv2 = Sequential(
            ReLU(inplace=True),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
        )
        self.shortcut = Sequential()
        if in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    padding=0,
                    bias=False,
                )
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class FlexCNN(Module):
    def __init__(self, output_dimension, layer_count=18, batch_normalization=True):
        super(FlexCNN, self).__init__()

        if layer_count < 2:
            raise ValueError(
                f"{layer_count} is not a valid layer count (at least 2 layers are required"
            )

        if batch_normalization:
            unit = ResidualUnitBN
        else:
            unit = ResidualUnit

        # possible channel counts for different unit layers
        channel_counts = [64, 128, 256, 512]

        # create input convolutional layer
        self.input_conv = Conv2d(
            3,
            channel_counts[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )

        # determine amount of units in each unit layer
        # don't count input convolution layer and output dense layer
        remaining_layer_count = layer_count - 2
        unit_counts = array([0] * len(channel_counts))
        unit_layers_count, final_convs_count = divmod(
            remaining_layer_count, unit.layer_count
        )
        for _ in range(unit_layers_count):
            # determine the unit layer with the least amount of units
            unit_counts[argmin(unit_counts)] += 1

        # create unit layers
        self.unit_layers = []
        for index, unit_count in enumerate(unit_counts):
            if unit_count == 0:
                break
            in_channels = channel_counts[0] if index == 0 else channel_counts[index - 1]
            out_channels = channel_counts[index]
            stride = 1 if index == 0 else 2
            units = [unit(in_channels, out_channels, stride)]
            for _ in range(unit_count - 1):
                units.append(unit(out_channels, out_channels, 1))
            self.unit_layers.append(Sequential(*units))
        self.unit_layers = Sequential(*self.unit_layers)

        # create final convolutional layers
        if unit_layers_count >= len(channel_counts):
            channels = channel_counts[-1]
        else:
            channels = channel_counts[unit_layers_count - 1]
        self.final_convs = []
        for _ in range(final_convs_count):
            conv = []
            if batch_normalization:
                conv.append(BatchNorm2d(channels))
            conv.append(ReLU(inplace=True))
            conv.append(
                Conv2d(
                    channels,
                    channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                    bias=False,
                )
            )
            self.final_convs.append(Sequential(*conv))
        self.final_convs = Sequential(*self.final_convs)

        # create output activation
        activation = []
        if batch_normalization:
            activation.append(BatchNorm2d(channels))
        activation.append(ReLU(inplace=True))

        self.output_activation = Sequential(*activation)

        # create output dense layer
        self.output_dense = Linear(channels, output_dimension)

    def forward(self, x):
        # apply input convolution layer
        y = self.input_conv(x)
        # apply unit layers
        y = self.unit_layers(y)
        # apply final convolution layers
        y = self.final_convs(y)
        # apply output activation layer
        y = self.output_activation(y)
        # average output activations over image dimensions
        y = avg_pool2d(y, y.size()[2:4])
        # flatten activations
        y = y.view(y.size(0), -1)
        # apply output dense layer
        y = self.output_dense(y)

        return y
