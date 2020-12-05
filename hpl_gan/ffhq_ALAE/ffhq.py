import torch
import torchvision
import math

"""
Loads full 1024x1024 FFHQ images 
"""


def get_ffhq_train_data(batch_size=64, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(
        root=store_location, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=4
    )

    return train_loader


def get_ffhq_test_data(batch_size=64, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test_dataset = torchvision.datasets.ImageFolder(
        root=store_location, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=4
    )

    return test_loader