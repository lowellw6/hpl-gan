import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm
import cv2


def save_reconstructions(name, image_tensors, reconstructions):
    reconstructions_shaped = reconstructions.view(-1, 1, 28, 28).to("cpu")
    
    results = create_compare_grid(image_tensors, reconstructions_shaped)
    results = results.detach().numpy()
    results *= 255.

    cv2.imwrite(name+".png", results)


def create_compare_grid(imgs, reconstructions, num_rows=4):
    imgs = torch.split(imgs, 1)
    reconstructions = torch.split(reconstructions, 1)

    assert len(imgs) % num_rows == 0
    row_len = len(imgs) / num_rows

    comp_rows = []
    for row in range(num_rows):
        start = int(row * row_len)
        stop = int((row + 1) * row_len)
        comp_imgs = torch.cat(imgs[start:stop], dim=3)
        comp_reconstr = torch.cat(reconstructions[start:stop], dim=3)
        comp_joint = torch.cat([comp_imgs, comp_reconstr], dim=2)
        comp_rows.append(comp_joint)

    return torch.cat(comp_rows, dim=2).squeeze()


def save_images(name, images):
    images_shaped = images.view(-1, 1, 28, 28).to("cpu")
    
    results = create_grid(images_shaped)
    results = results.detach().numpy()
    results *= 255.

    cv2.imwrite(name+".png", results)


def create_grid(imgs, num_rows=4):
    imgs = torch.split(imgs, 1)

    assert len(imgs) % num_rows == 0
    row_len = len(imgs) / num_rows

    comp_rows = []
    for row in range(num_rows):
        start = int(row * row_len)
        stop = int((row + 1) * row_len)
        comp_imgs = torch.cat(imgs[start:stop], dim=3)
        comp_rows.append(comp_imgs)

    return torch.cat(comp_rows, dim=2).squeeze()