from typing import List

import torch
from torch import Tensor
from torchvision.transforms import Resize, Normalize
from torch.nn import Sequential

from multi_attribute_classifier import MultiAttributeClassifier
from efficientnet_pytorch import EfficientNet as EfficientNetBase


class EfficientNet(MultiAttributeClassifier):
    """
    EfficientNet is a scalable network for multi attribute prediction.

    See https://doi.org/10.48550/arXiv.1905.11946 for more information on the original architecture.
    """
    def __init__(
        self,
        attribute_sizes: List[int],
        attribute_class_weights: List[Tensor],
        b: int = 1,
        base_out_filter_count: int = 1000,
    ):
        """
        Creates a new EfficientNet instance.

        :param attribute_sizes: List[int] contains the class counts for each predicted attribute
        :param attribute_class_weights: List[Tensor] contains a Tensor[attribute_class_count] for each predicted
            attribute that stores the class weighting coefficients (in the range [0, 1] with a sum of 1)
        :param b: Integer coefficient that scales the base network size. Possible values are in the range [0, 7]
        :param base_out_filter_count: Base network output filter count
        """
        MultiAttributeClassifier.__init__(
            self,
            attribute_sizes,
            attribute_class_weights,
            multi_output_in_filter_count=base_out_filter_count,
        )

        self.b = b
        image_size = EfficientNetBase.get_image_size(f"efficientnet-b{b}")
        self.add_module(
            "image_transform",
            Sequential(
                Resize(image_size),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ),
        )
        self.add_module(
            "net", EfficientNetBase.from_pretrained(f"efficientnet-b{b}", num_classes=base_out_filter_count)
        )

    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feeds an input Tensor through the network.

        :param x: Input Tensor[N, 3, image_height, image_width] containing N images of arbitrary size
        :return: Tensor[N, base_out_filter_count] containing N feature vectors with base_out_filter_count elements
        """
        return self.net(self.image_transform(x))
