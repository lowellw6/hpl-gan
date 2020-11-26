import torch
import torchvision

### TODO adjust transform to resize to 128x128

def get_imagenet_train_data(batch_size=128, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageNet(
        root=store_location, split="train", transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return train_loader


def get_imagenet_test_data(batch_size=32, num_workers=4, store_location="."):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test_dataset = torchvision.datasets.ImageNet(
        root=store_location, split="val", transform=transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader