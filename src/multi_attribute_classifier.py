from abc import ABC, abstractmethod

import torch
from torch import flatten, tensor, sigmoid, softmax, stack, Tensor
from torch.nn import Linear, Module
from typing import List, Tuple


class MultiAttributeClassifier(ABC, Module):
    """
    MultiAttributeClassifier is an abstract base class that uses the output feature vectors of feed forward networks to
    predict multiple attributes at once.

    Deriving classes need to implement the :meth:`final_layer_output` method and initialise this base class at
    construction.
    """
    def __init__(
        self, attribute_sizes: List[int], attribute_class_weights: List[Tensor], multi_output_in_filter_count: int
    ):
        """
        Initialises the MultiAttributeClassifier.

        :param attribute_sizes: List[int] contains the class counts for each predicted attribute
        :param attribute_class_weights: List[Tensor] contains a Tensor[attribute_class_count] for each predicted
            attribute that stores the class weighting coefficients (in the range [0, 1] with a sum of 1)
        :param multi_output_in_filter_count: Dimensionality of the output feature vector from the deriving network class
        """
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
        """
        Feeds an input Tensor through the deriving network instance.

        Abstract method that needs to be implemented by the deriving class.

        :param x: Input Tensor[N, 3, image_height, image_width] containing N images of arbitrary size
        :return: Tensor[N, base_out_filter_count] containing N feature vectors with base_out_filter_count elements
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Feeds an input Tensor through the wrapped network.

        :param x: Input Tensor[N, 3, image_height, image_width] containing N images of arbitrary size
        :return: List[Tensor] containing the multi output class logits stored in a
            Tensor[N, attribute_count(, attribute_class_count)] for each unique attribute size (number of classes).
            Each tensor contains the predicted logits for each of the N samples, attributes of the corresponding size
            and classes of the corresponding attribute. The last dimension is omitted if an attribute is binary.
        """
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
        """
        Computes indices of an attribute in the multi output class logits from the :meth:`forward` method.

        :param attribute_index: Prediction attribute index (corresponds to the order of attributes in attribute_sizes)
        :return: int list index of the Tensor within the wrapped model output,
                 int attribute dimension index of the predicted attribute with this Tensor
        """
        multi_output_index = self.inverse_attribute_size_indices[attribute_index].item()
        attribute_size = self.attribute_sizes[attribute_index]
        previous_attribute_sizes = self.attribute_sizes[0:attribute_index]
        output_index = previous_attribute_sizes.eq(attribute_size).sum().item()
        return multi_output_index, output_index

    def multi_attribute_predictions(self, multi_output_class_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the attribute class labels from the multi output class logits from the :meth:`forward` method.

        :param multi_output_class_logits: List[Tensor] returned from :meth:`forward` method containing predicted logits
        :return: Tensor[N, attribute_prediction_count] storing the most likely class label of each sample and attribute
        """
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
        """
        Computes class probabilities of an attribute from the multi output class logits from the :meth:`forward` method.

        :param multi_output_class_logits: List[Tensor] returned from :meth:`forward` method containing predicted logits
        :param attribute_index: Prediction attribute index (corresponds to the order of attributes in attribute_sizes)
        :return: Tensor[N, class_count] containing the probabilities for each sample and attribute class
        """
        multi_output_index, output_index = self.multi_output_indices(attribute_index)
        class_logits = multi_output_class_logits[multi_output_index][..., output_index]
        if class_logits.dim() == 1:
            class_1_probabilities = sigmoid(class_logits)
            class_0_probabilities = 1 - class_1_probabilities
            _attribute_probabilities = stack([class_0_probabilities, class_1_probabilities], dim=1)
        else:
            _attribute_probabilities = softmax(class_logits, dim=1)
        return _attribute_probabilities
