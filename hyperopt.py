import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import optuna
from torch import no_grad, tensor
from torchmetrics import JaccardIndex

from experiment import fair_attribute_prediction_experiment
from losses.fair_losses import fair_losses
from util import create_dataset, create_dataloader, get_device, create_model

from joblib import dump, load


if __name__ == "__main__":
    assert len(sys.argv) == 3
    args_root_dir_path = Path(sys.argv[1])
    relative_args_file_path = Path(sys.argv[2])
    absolute_args_file_path = args_root_dir_path / relative_args_file_path

    parser = ArgumentParser(fromfile_prefix_chars="+")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_count", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning_rate_scheduler",
        default="none",
        choices=["none", "reduce_lr_on_plateau"],
    )
    parser.add_argument("--reduce_lr_on_plateau_factor", type=float, default=0.5)
    parser.add_argument("--reduce_lr_on_plateau_patience", type=int, default=5)
    parser.add_argument("--dataset", default="celeba", choices=["utkface", "celeba", "siim_isic_melanoma"])
    parser.add_argument(
        "--model",
        default="slimcnn",
        choices=[
            "slimcnn",
            "simplecnn",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ],
    )
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--sensitive_attribute_index", required=True, type=int)
    parser.add_argument("--target_attribute_index", required=True, type=int)
    parser.add_argument("--pretrained_model")
    parser.add_argument("--class_weights", default="none", choices=["none", "balanced", "ins", "isns", "ens"])
    parser.add_argument("--ens_beta", type=float, default=0.99)
    parser.add_argument("--number_of_trials", type=int, default=100)
    parser.add_argument("--fair_loss_weight_min", type=float, default=1.0e-7)
    parser.add_argument("--fair_loss_weight_max", type=float, default=1.0e2)
    parser.add_argument(
        "--fair_loss_type",
        default="intersection_over_union_conditioned",
        choices=fair_losses.keys(),
    )
    parser.add_argument("--resume_hpo_study")

    arguments = parser.parse_args([f"+{absolute_args_file_path}"])
    parameters = deepcopy(vars(arguments))
    del parameters["fair_loss_weight_min"]
    del parameters["fair_loss_weight_max"]
    del parameters["number_of_trials"]

    train_dataset = create_dataset(parameters, split_name="train")
    valid_dataset = create_dataset(parameters, split_name="valid")

    valid_dataloader = create_dataloader(parameters, valid_dataset)

    if arguments.resume_hpo_study is None:
        study = optuna.create_study(direction="minimize")
        start_date = datetime.utcnow().isoformat()
        study.set_user_attr("start_date", start_date)
    else:
        assert Path(arguments.resume_hpo_study).is_file()
        study = load(arguments.resume_hpo_study)
        start_date = study.user_attrs["start_date"]
    performed_trial_count = len(study.trials)

    for i in range(performed_trial_count, arguments.number_of_trials):
        trial = study.ask()

        parameters["fair_loss_weight"] = trial.suggest_loguniform(
            "fair_loss_weight", arguments.fair_loss_weight_min, arguments.fair_loss_weight_max
        )

        experiment_name_parts = [
            args_root_dir_path.name,
            *relative_args_file_path.parts[:-1],
            relative_args_file_path.stem,
        ]
        experiment_name = "-".join(experiment_name_parts) + f"-HPO-#{i:04}"

        experiment, best_model_state, final_model_state = fair_attribute_prediction_experiment(
            parameters, experiment_name
        )

        target_attribute = train_dataset.attribute(parameters["target_attribute_index"])
        target_prediction_attribute_index = train_dataset.prediction_attribute_indices.index(target_attribute.index)

        sensitive_attribute = train_dataset.attribute(parameters["sensitive_attribute_index"])
        attribute_ious = [
            JaccardIndex(num_classes=target_attribute.size).to(get_device()) for _ in range(sensitive_attribute.size)
        ]
        model = create_model(parameters, train_dataset)
        model.load_state_dict(final_model_state["model_state_dict"])
        model.eval()

        with no_grad():
            for images, attributes, indices in valid_dataloader:
                images, attributes = images.to(get_device()), attributes.to(get_device())

                target_attribute.targets = attributes[:, target_attribute.index]
                sensitive_attribute.targets = attributes[:, sensitive_attribute.index]

                multi_output_class_logits = model(images)
                target_attribute.class_probabilities = model.module.attribute_class_probabilities(
                    multi_output_class_logits, target_prediction_attribute_index
                )

                target_attribute.predictions = model.module.multi_attribute_predictions(multi_output_class_logits)[
                    :, target_prediction_attribute_index
                ]

                for sensitive_attribute_class_b in range(sensitive_attribute.size):
                    from_class_b = sensitive_attribute.targets.eq(sensitive_attribute_class_b)
                    if from_class_b.sum() == 0:
                        continue
                    attribute_ious[sensitive_attribute_class_b](
                        target_attribute.predictions[from_class_b], target_attribute.targets[from_class_b]
                    )

            attribute_ious = [attribute_iou.compute().item() for attribute_iou in attribute_ious]

        attribute_iou_std = tensor(attribute_ious).std(dim=0).item()
        study.tell(trial, attribute_iou_std)
        study_state_dir = (
            Path("experiments") / "results" / ("-".join(experiment_name_parts) + "-HPO") / start_date
        )
        study_state_dir.mkdir(parents=True, exist_ok=True)
        study_state_file_path = study_state_dir / "study.pkl"
        dump(study, study_state_file_path)

        best_fair_loss_weight = study.best_trial.suggest_loguniform(
            "fair_loss_weight", arguments.fair_loss_weight_min, arguments.fair_loss_weight_max
        )
        best_attribute_iou_std = study.best_trial.value

        print(f"HPO-#{i:04}: λ={parameters['fair_loss_weight']:.3E} IoU-STD={attribute_iou_std:.3E}")
        print(f"HPO-#Best: λ={best_fair_loss_weight:.3E} IoU-STD={best_attribute_iou_std:.3E}")
