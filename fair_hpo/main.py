from os.path import join as join_path

import torch
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from numpy.random import RandomState
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from data.UTKFace import load_utkface
from evaluation.Evaluation import classifier_accuracy
from models.FlexCNN import FlexCNN
from training.Training import train_classifier


def train_model(hyperparameter_config):
    layer_count = hyperparameter_config["layer_count"]
    batch_normalization = hyperparameter_config["batch_normalization"] == "true"
    max_learning_rate = hyperparameter_config["max_learning_rate"]
    base_learning_rate_factor = hyperparameter_config["base_learning_rate_factor"]
    base_learning_rate = base_learning_rate_factor * max_learning_rate
    momentum = hyperparameter_config["momentum"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlexCNN(layer_count=layer_count, batch_normalization=batch_normalization).to(device)

    optimizer = SGD(model.parameters(), lr=max_learning_rate, momentum=momentum)
    lr_scheduler = CyclicLR(optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate)
    criterion = CrossEntropyLoss()

    for epoch in range(1, epoch_count + 1):
        train_classifier(model, train_dataloader, optimizer, lr_scheduler, criterion, display_progress=True)

    return model


def evaluate_cost(hyperparameter_config):
    trained_model = train_model(hyperparameter_config)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                       pin_memory=True)
    accuracy = classifier_accuracy(trained_model, validation_dataloader)
    cost = 1 - accuracy
    return cost


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    data_directory = "/home/tobias/data/discoret"
    utkface_directory = join_path(data_directory, "UTKFace")

    num_workers = 4
    batch_size = 32
    epoch_count = 10

    train_dataset, validation_dataset, test_dataset = load_utkface(image_directory_path=utkface_directory)

    hyperparameter_config_space = ConfigurationSpace()
    layer_count_hyperparameter = UniformIntegerHyperparameter("layer_count", 3, 24, default_value=10)
    batch_normalization_hyperparameter = CategoricalHyperparameter("batch_normalization", ["true", "false"],
                                                                   default_value="true")
    max_learning_rate_hyperparameter = UniformFloatHyperparameter("max_learning_rate", 0.0001, 10.0, default_value=0.5,
                                                                  log=True)
    base_learning_rate_factor_hyperparameter = UniformFloatHyperparameter("base_learning_rate_factor", 0.0001, 1.0,
                                                                          default_value=0.01)
    momentum_hyperparameter = UniformFloatHyperparameter("momentum", 0.0, 1.0, default_value=0.0)
    hyperparameter_config_space.add_hyperparameters(
        [layer_count_hyperparameter, batch_normalization_hyperparameter, max_learning_rate_hyperparameter,
         base_learning_rate_factor_hyperparameter,
         momentum_hyperparameter])

    scenario = Scenario(
        {"run_obj": "quality", "runcount-limit": 10, "cs": hyperparameter_config_space, "deterministic": "false"})

    smac = SMAC4HPO(scenario=scenario, rng=RandomState(42), tae_runner=evaluate_cost)

    incumbent_hyperparameter_config = smac.optimize()

    incumbent_model = train_model(incumbent_hyperparameter_config)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 pin_memory=True)
    test_accuracy = classifier_accuracy(incumbent_model, test_dataloader)
    print("Incumbent hyperparameter configuration: ", incumbent_hyperparameter_config)
    print("Test accuracy: ", test_accuracy)
