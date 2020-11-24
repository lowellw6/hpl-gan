import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm
import cv2

from model import AE, LatentGeneratorMLP, LatentDiscriminatorMLP
from mnist import get_mnist_test_data
from util import save_reconstructions, save_images



def eval_autoencoder(state_dict_path):
    """
    Evaluate on held-out set
    Setting state_dict_path=None will evaluate random intialization
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_test_loader = get_mnist_test_data(store_location="./datasets", batch_size=1024)

    autoencoder = AE(input_shape=784, z_size=128).to(device)
    if state_dict_path:
        autoencoder.load_state_dict(torch.load(state_dict_path, map_location=device))
    autoencoder.eval()

    print(f"Evaluating on MNIST digits held-out set")
    loss = 0.
    for image_tensors, _ in tqdm(mnist_test_loader):
        input_batch = image_tensors.view(-1, 784).to(device)
        input_features = (input_batch * 2) - 1
        reconstructions = autoencoder(input_features)
        loss += F.mse_loss(reconstructions, input_features)
    
    loss /= len(mnist_test_loader)
    print(f"Held-out loss = {loss:.6f}")

    # Display reconstructions on first batch
    test_batch, _ = next(iter(mnist_test_loader))
    test_input = (test_batch * 2) - 1
    reconstructions = autoencoder(test_input)
    reconstructions = (reconstructions + 1) * 0.5
    save_reconstructions(f"AE-grid-eval", test_batch, reconstructions)


def eval_hpl(ae_sd_path, zgen_sd_path, zdis_sd_path):
    """
    Evaluate on held-out set
    Setting *_sd_path=None will evaluate random intialization for that net
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")
    
    mnist_test_loader = get_mnist_test_data(store_location="./datasets")
    test_batch, _ = next(iter(mnist_test_loader))  # Used for visual checkpoints of progress

    z_size = 128

    autoencoder = AE(input_shape=784, z_size=z_size).to(device)
    if ae_sd_path:
        autoencoder.load_state_dict(torch.load(ae_sd_path, map_location=device))

    zgen = LatentGeneratorMLP(z_size, 32).to(device)
    if zgen_sd_path:
        zgen.load_state_dict(torch.load(zgen_sd_path, map_location=device))

    zdis = LatentDiscriminatorMLP(z_size, 32).to(device)
    if zdis_sd_path:
        zdis.load_state_dict(torch.load(zdis_sd_path, map_location=device))

    print(f"Evaluating HPL on MNIST digits held-out set")
    autoencoder.eval()
    zgen.eval()
    zdis.eval()
    Dloss = 0
    Gloss = 0
    for image_tensors, _ in tqdm(mnist_test_loader):
        ones_target = torch.ones(len(image_tensors), 1).to(device)
        zeros_target = torch.zeros(len(image_tensors), 1).to(device)

        # Compute discriminator loss with real and fake latent codes
        input_features = image_tensors.view(-1, 784).to(device)
        input_features = (input_features * 2) - 1
        real_codes = autoencoder.encode(input_features)
        real_codes = real_codes.detach()

        real_scores = zdis(real_codes)
        real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)  # Smoothed "real" label

        prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
        fake_codes = zgen(prior)
        fake_scores = zdis(fake_codes.detach())
        fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

        zdis_loss = real_loss + fake_loss
        Dloss += zdis_loss.item()

        # Compute generator loss for maximizing fooling of zdis
        prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
        fake_codes = zgen(prior)
        fake_scores = zdis(fake_codes)
        zgen_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)
        Gloss += zgen_loss.item()

    Dloss /= len(mnist_test_loader)
    Gloss /= len(mnist_test_loader)

    print(f"Held-out D-loss = {Dloss:.6f}, G-loss = {Gloss:.6f}")

    # Display reconstructions on first batch
    prior = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
    testZ = zgen(prior)
    testX = (autoencoder.decode(testZ) + 1) * 0.5
    save_images(f"HPL-grid-eval", testX)



if __name__ == "__main__":
    eval_hpl("./models/ae20.pt", "./models/zgen100.pt", "./models/zdis100.pt")