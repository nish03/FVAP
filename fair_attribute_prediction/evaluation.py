from typing import Dict

import comet_ml
from numpy import array
from sklearn.metrics import roc_auc_score, confusion_matrix

import torch.utils.data
from torch import no_grad
from torch.cuda import empty_cache

from util import get_device


def evaluate_classifier(
    model: torch.nn.Module,
    model_state: Dict,
    dataloader: torch.utils.data.DataLoader,
    parameters: Dict,
    experiment: comet_ml.Experiment,
) -> (Dict[str, float], torch.Tensor):
    empty_cache()

    model_state_dict = model_state["model_state_dict"]
    for key, value in model_state_dict.items():
        model_state_dict[key] = value.to(get_device())
    model.load_state_dict(model_state_dict)

    target_attribute = dataloader.dataset.attribute(parameters["target_attribute_index"])
    target_attribute.targets = []
    target_attribute.class_probabilities = []
    target_attribute.predictions = []

    prediction_attribute_indices = dataloader.dataset.prediction_attribute_indices

    scores = dict()

    with experiment.validate():
        model.eval()
        with no_grad():
            for images, attributes in dataloader:
                images, attributes = images.to(get_device()), attributes.to(get_device())

                multi_output_class_logits = model(images)
                target_attribute.targets += attributes[:, target_attribute.index].tolist()
                target_prediction_attribute_index = prediction_attribute_indices.index(target_attribute.index)
                target_attribute.class_probabilities += model.module.attribute_class_probabilities(
                    multi_output_class_logits, target_prediction_attribute_index
                ).tolist()
                target_attribute.predictions += model.module.multi_attribute_predictions(multi_output_class_logits)[
                    :, target_prediction_attribute_index
                ].tolist()

    if target_attribute.size == 2:
        scores["roc_auc"] = roc_auc_score(target_attribute.targets, array(target_attribute.class_probabilities)[:, 1])
    else:
        scores["roc_auc"] = roc_auc_score(
            target_attribute.targets,
            target_attribute.class_probabilities,
            multi_class="ovo",
            average="weighted",
        )

    for key, value in model_state_dict.items():
        model_state_dict[key] = value.cpu()

    _confusion_matrix = confusion_matrix(target_attribute.targets, target_attribute.predictions)
    return scores, _confusion_matrix
