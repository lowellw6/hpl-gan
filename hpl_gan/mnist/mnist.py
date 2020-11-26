import torch
import torchvision

"""
Adapted from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""


def get_mnist_train_data(batch_size=128, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root=store_location, train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return train_loader


def get_mnist_test_data(batch_size=32, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test_dataset = torchvision.datasets.MNIST(
        root=store_location, train=False, transform=transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader

