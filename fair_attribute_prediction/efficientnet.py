import torch
from torchvision.transforms import Resize, Normalize
from torch.nn import Sequential

from multi_attribute_classifier import MultiAttributeClassifier
from efficientnet_pytorch import EfficientNet as EfficientNetBase


class EfficientNet(MultiAttributeClassifier):
    def __init__(
        self,
        b=3,
        base_out_filter_count=1000,
        attribute_sizes=None,
        attribute_class_weights=None,
    ):
        MultiAttributeClassifier.__init__(
            self,
            attribute_sizes,
            attribute_class_weights,
            multi_output_in_filter_count=base_out_filter_count,
        )

        self.b = b
        self.add_module(
            "image_transform",
            Sequential(
                Resize(224),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ),
        )
        self.add_module(
            "net", EfficientNetBase.from_pretrained(f"efficientnet-b{b}", num_classes=base_out_filter_count)
        )

    def final_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.image_transform(x))
