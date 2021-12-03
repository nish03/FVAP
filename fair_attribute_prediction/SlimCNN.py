from collections import defaultdict
from itertools import chain

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
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.nn.init import xavier_uniform_


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

class SlimCNN(Module):
    def __init__(
        self,
        squeeze_filter_counts=None,
        variable_class_counts=None,
        criterion_l2_weight=0.0001,
    ):
        super().__init__()

        if squeeze_filter_counts is None:
            squeeze_filter_counts = [16, 32, 48, 64]
        self.squeeze_filter_counts = tensor(squeeze_filter_counts)
        self.variable_class_counts = tensor(variable_class_counts)
        (
            self.class_counts,
            self.variable_class_count_indices,
            self.variable_counts,
        ) = self.variable_class_counts.unique(
            sorted=True, return_inverse=True, return_counts=True
        )
        self.criterion_l2_weight = criterion_l2_weight

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
        self.add_fully_connected_layer()
        self.weight_regularisation_parameters = [
            [parameter for parameter in module.parameters()]
            for module in [
                self.layer_3_slim_module.layer_3_dw_sep_3x3_conv,
                self.layer_5_slim_module.layer_3_dw_sep_3x3_conv,
                self.layer_7_slim_module.layer_3_dw_sep_3x3_conv,
                self.layer_9_slim_module.layer_3_dw_sep_3x3_conv,
            ]
        ]
        self.weight_regularisation_parameters = list(
            chain.from_iterable(self.weight_regularisation_parameters)
        )

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

    def add_fully_connected_layer(self):
        self.layer_count += 1
        fully_connected = Module()
        for class_count, variable_count in zip(self.class_counts, self.variable_counts):
            output_module = Linear(
                self.squeeze_filter_counts[self.slim_module_count - 1] * 3,
                (1 if class_count == 2 else class_count) * variable_count,
            )
            fully_connected.add_module(
                "layer_0_"
                + (
                    "binary"
                    if class_count == 2
                    else f"categorical_{class_count.item()}"
                ),
                output_module,
            )
        self.add_module(f"layer_{self.layer_count}_fully_connected", fully_connected)

    def forward(self, x):
        x = self.layer_1_convolution(x)
        x = self.layer_2_max_pooling(x)
        x = self.layer_3_slim_module(x)
        x = self.layer_4_max_pooling(x)
        x = self.layer_5_slim_module(x)
        x = self.layer_6_max_pooling(x)
        x = self.layer_7_slim_module(x)
        x = self.layer_8_max_pooling(x)
        x = self.layer_9_slim_module(x)
        x = self.layer_10_max_pooling(x)
        x = self.layer_11_global_pooling(x)
        x = [
            output_module(flatten(x, 1))
            if class_count == 2
            else output_module(flatten(x, 1)).reshape(-1, class_count, variable_count)
            for class_count, variable_count, output_module in zip(
                self.class_counts,
                self.variable_counts,
                self.layer_12_fully_connected.children(),
            )
        ]
        return x

    def predict(self, outputs):
        output_predictions = []
        for class_score_predictions in outputs:
            output_predictions.append(
                (class_score_predictions > 0.0).long()
                if class_score_predictions.dim() == 2
                else class_score_predictions.argmax(dim=1)
            )
        class_index_predictions = []
        output_prediction_indices = defaultdict(int)
        for variable_class_count_index, output_index in enumerate(
            self.variable_class_count_indices
        ):
            variable_class_count = self.variable_class_counts[
                variable_class_count_index
            ]
            output_prediction_index = output_prediction_indices[
                variable_class_count.item()
            ]
            class_index_predictions.append(
                output_predictions[output_index][:, output_prediction_index]
            )
            output_prediction_indices[variable_class_count.item()] += 1
        class_index_predictions = stack(class_index_predictions, dim=1)
        return class_index_predictions

    def criterion(self, outputs, class_index_targets):
        loss_terms = defaultdict(float)
        for class_score_predictions in outputs:
            if class_score_predictions.dim() == 2:
                class_count = 2
                loss_terms["binary"] += binary_cross_entropy_with_logits(
                    class_score_predictions,
                    class_index_targets[
                        :, self.variable_class_counts == class_count
                    ].float(),
                    reduction="sum",
                )
            else:
                class_count = class_score_predictions.shape[1]
                loss_terms[f"categorical_{class_count}"] += cross_entropy(
                    class_score_predictions,
                    class_index_targets[:, self.variable_class_counts == class_count],
                    reduction="sum",
                )
        for loss_part_name in loss_terms:
            loss_terms[loss_part_name] /= (
                class_index_targets.shape[0] * class_index_targets.shape[1]
            )

        loss_terms["l2_penalty"] = self.criterion_l2_weight * sum(
            [
                parameter.pow(2).sum()
                for parameter in self.weight_regularisation_parameters
            ]
        )
        loss = sum(loss_terms.values())
        return loss, loss_terms
