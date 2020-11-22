import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm
import cv2

from model import AE
from mnist import get_mnist_test_data



def eval_autoencoder(state_dict_path):
    """
    Evaluate on held-out set
    Setting state_dict_path=None will evaluate random intialization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_test_loader = get_mnist_test_data(store_location="./datasets")

    autoencoder = AE(input_shape=784).to(device)
    if state_dict_path:
        autoencoder.load_state_dict(torch.load(state_dict_path, map_location=device))

    print(f"Evaluating on MNIST digits held-out set")
    loss = 0.
    for image_tensors, _ in tqdm(mnist_test_loader):
        input_features = image_tensors.view(-1, 784).to(device)
        reconstructions = autoencoder(input_features)
        loss += F.mse_loss(reconstructions, input_features)
    
    loss /= len(mnist_test_loader)
    print(f"Held-out loss = {loss:.6f}")

    # Display reconstructions on first batch
    first_batch, _ = next(iter(mnist_test_loader))
    save_reconstructions("AE-grid", autoencoder, device, first_batch)


def save_reconstructions(name, model, device, image_tensors):
    input_features = image_tensors.view(-1, 784).to(device)
    reconstructions = model(input_features)
    
    reconstructions_shaped = reconstructions.view(-1, 1, 28, 28).to("cpu")
    
    results = create_compare_grid(image_tensors, reconstructions_shaped)
    results = results.detach().numpy()
    results *= 255.

    cv2.imwrite(name+".png", results)


def create_compare_grid(imgs, reconstructions):
    imgs = torch.split(imgs, 1)
    reconstructions = torch.split(reconstructions, 1)

    assert len(imgs) % 4 == 0
    row_len = len(imgs) / 4

    comp_rows = []
    for row in range(4):
        start = int(row * row_len)
        stop = int((row + 1) * row_len)
        comp_imgs = torch.cat(imgs[start:stop], dim=3)
        comp_reconstr = torch.cat(reconstructions[start:stop], dim=3)
        comp_joint = torch.cat([comp_imgs, comp_reconstr], dim=2)
        comp_rows.append(comp_joint)

    return torch.cat(comp_rows, dim=2).squeeze()


def save_images(name, images, device):
    images_shaped = images.view(-1, 1, 28, 28).to("cpu")
    
    results = create_grid(images_shaped)
    results = results.detach().numpy()
    results *= 255.

    cv2.imwrite(name+".png", results)


def create_grid(imgs):
    imgs = torch.split(imgs, 1)

    assert len(imgs) % 4 == 0
    row_len = len(imgs) / 4

    comp_rows = []
    for row in range(4):
        start = int(row * row_len)
        stop = int((row + 1) * row_len)
        comp_imgs = torch.cat(imgs[start:stop], dim=3)
        comp_rows.append(comp_imgs)

    return torch.cat(comp_rows, dim=2).squeeze()



if __name__ == "__main__":
    eval_autoencoder("./models/ae20.pt")