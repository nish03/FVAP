import torch
from torch import (
    cat,
    load,
    float32,
    cuda
)
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Resize
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from sklearn import svm
from model.FlexVAE import FlexVAE
from data.Util import load_dataset

from sys import argv
from argparse import ArgumentParser
arg_parser = ArgumentParser(
    description="classification of sensitive attributes using VAE representations",
    fromfile_prefix_chars="+",
)
arg_parser.add_argument(
    "--batch-size",
    default=144,
    type=int,
    required=False,
    help="Batch size for loading the dataset",
)
arg_parser.add_argument(
    "--image-size",
    default=64,
    type=int,
    required=False,
    help="Image size for loading the dataset",
)
arg_parser.add_argument(
    "--dataset",
    default="CelebA",
    help="Dataset for training the generative model",
    choices=["UTKFace", "CelebA", "LFWA+", "FairFace"],
    required=False,
)
arg_parser.add_argument(
    "--dataset-dir",
    default="/srv/nfs/data/mengze/vae",
    help="Directory for loading the dataset",
    required=False,
)
arg_parser.add_argument(
    "--sensitive-attribute",
    type=int,
    default=0,
    required=False,
    help="Index of the sensitive attribute for fairness optimization",
)
arg_parser.add_argument(
    "--output-dir",
    default=".",
    required=False,
    help="Output directory for storing log files, save states and HPO runs",
)
arg_parser.add_argument(
    "--vae-trained",
    default="/srv/nfs/data/mengze/vae/bohb/recon0901.pt",
    help="pre-trained VAE model",
    required=False,
)
args = arg_parser.parse_args(argv[1:])

if cuda.is_available():
    device = "cuda"
    device_count = cuda.device_count()

else:
    device = "cpu "
    device_count = 1
num_workers = device_count * 4
if cudnn.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False
################################################
#device = "cpu"
#num_workers = 0
################################################

transform = Compose(
    [ConvertImageDtype(float32), Resize(args.image_size), Lambda(lambda x: 2 * x - 1)]
)

target_transform = Lambda(lambda x: x[args.sensitive_attribute])
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(16593))
validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(2880))
(train_dataset, validation_dataset, test_dataset), dataset_class, dataset_dir = load_dataset(
    dataset_name=args.dataset,
    dataset_dir=args.dataset_dir,
    transform=transform,
    target_transform=target_transform,
    in_memory=True,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    sampler=train_sampler
)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=args.batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    sampler=validation_sampler
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    sampler=train_sampler
)

print("Starting to transform data into representation vectors")
model_state = load(args.vae_trained, map_location=torch.device(device))

hyperparameters = model_state['hyper_params']
model_state_dict = model_state["model_state_dict"]
vae_model = FlexVAE(
    args.image_size,
    hyperparameters.latent_dimension_count,
    hyperparameters.hidden_layer_count,
    hyperparameters.vae_loss_gamma,
    hyperparameters.C_max,
    hyperparameters.C_stop_iteration,
    hyperparameters.reconstruction_loss,
    hyperparameters.reconstruction_loss_args,
    hyperparameters.reconstruction_loss_label_weights, #[0.05, 0.8, 0.05, 0.05, 0.05],
    hyperparameters.kld_loss_label_weights, #[0.2, 0.2, 0.2, 0.2, 0.2],
    hyperparameters.weighted_average_type, #"IndividualLosses"
)

vae_model.load_state_dict(model_state_dict)
vae_model = vae_model.to(device)
vae_model.eval()


#images = []
targets = []
representations = []
# reconstructions = []
for image_batch, target_batch in train_dataloader:
    image_batch = image_batch.to(device)
    mu, var = vae_model.encode(image_batch)
    representation = vae_model.reparameterize(mu, var).detach().cpu()
    representations.append(representation)
    # reconstruction_batch = vae_model.reconstruct(image_batch.to(device))
    # images.append(image_batch)
    targets.append(target_batch)
    # reconstructions.append(reconstruction_batch)
# images = cat(images)
targets = cat(targets)
# reconstructions = cat(reconstructions)
representations = cat(representations)

# validation_images = []
validation_targets = []
# validation_reconstructions = []
validation_representations = []
for image_batch, target_batch in validation_dataloader:
    image_batch = image_batch.to(device)
    mu, var = vae_model.encode(image_batch)
    validation_representation = vae_model.reparameterize(mu, var).detach().cpu()
    validation_representations.append(validation_representation)
    # reconstruction_batch = vae_model.reconstruct(image_batch.to(device))
   #  validation_images.append(image_batch)
    validation_targets.append(target_batch)
    # validation_reconstructions.append(reconstruction_batch)
# validation_images = cat(validation_images)
validation_targets = cat(validation_targets)
# validation_reconstructions = cat(validation_reconstructions)
validation_representations = cat(validation_representations)
print("Starting to do classification")
# from sklearn.model_selection import GridSearchCV
# parameters = {'C': np.logspace(-2,2,5), 'kernel': ('linear', 'rbf', 'poly', 'sigmoid')}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(representations, targets)
#C = [0.01, 0.1, 1, 10, 100]
C = [1]
for c in C:
    print("c", c)
    clf_svm = svm.SVC(C=c, kernel='sigmoid').fit(representations, targets)
    score_svm = clf_svm.score(validation_representations, validation_targets)
    print(f"overall acc: {score_svm:.4f} ")
    predictions_svm = clf_svm.predict(validation_representations)
    predictions_svm = torch.as_tensor(predictions_svm)
    target_classes = np.unique(validation_targets)
    precisions = []
    if torch.is_tensor(validation_targets) == False:
        print("this is not a tensor")
        validation_targets = torch.as_tensor(validation_targets)
    for target_class in target_classes:
        TP_count = torch.sum((predictions_svm == validation_targets) & (validation_targets == target_class)).item()
        precision = (TP_count/torch.sum(validation_targets == target_class)).item()
        precisions.append(precision)
    print("precision for each sensitive attribute", precisions)
