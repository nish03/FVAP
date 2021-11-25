import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 1

