import torch
import torchvision
import math

"""
Currently supports loading 128x128 thumbnails (rather than main dataset of 1024x1024)
"""


def get_ffhq_train_data(batch_size=64, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((118, 118))])

    train_dataset = torchvision.datasets.ImageFolder(
        root=store_location, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return train_loader


def get_ffhq_test_data(batch_size=16, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((118, 118))])

    test_dataset = torchvision.datasets.ImageFolder(
        root=store_location, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader