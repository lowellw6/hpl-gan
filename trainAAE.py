"""
Training basic Adversarial AutoEncoder (AAE)
https://arxiv.org/pdf/1511.05644.pdf
"""

import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm

from model import AE, LatentDiscriminatorMLP
from mnist import get_mnist_train_data, get_mnist_test_data
from util import save_images


def train_adversarial_autoencoder(epochs):
    """
    Note "real_codes" and "fake_codes" are reversed compared to HPL
    real --> from prior ~ Uniform [0, 1)
    fake --> from AAE encoder (generator)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_train_loader = get_mnist_train_data(store_location="./datasets")
    
    mnist_test_loader = get_mnist_test_data(store_location="./datasets")
    test_batch, _ = next(iter(mnist_test_loader))  # Used for visual checkpoints of progress

    autoencoder = AE(input_shape=784).to(device)
    ae_opt = optim.Adam(autoencoder.parameters(), lr=1e-3)

    zdis = LatentDiscriminatorMLP(128).to(device)
    zdis_opt = optim.Adam(zdis.parameters(), lr=1e-3)

    with torch.no_grad():
        testZ = torch.rand(len(test_batch), 128).to(device)  # Directly from random uniform distribution [0, 1)
        testX = autoencoder.decode(testZ)
    save_images(f"AAE-grid-0", testX)

    print(f"Training AAE for {epochs} epochs on MNIST digits")
    for epoch in range(epochs):
        Dloss = 0
        Gloss = 0
        AEloss = 0
        for image_tensors, _ in tqdm(mnist_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            input_features = image_tensors.view(-1, 784).to(device)
            fake_codes = autoencoder.encode(input_features)

            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy(fake_scores, zeros_target)

            real_codes = torch.rand(len(image_tensors), 128).to(device)
            real_scores = zdis(real_codes.detach())
            real_loss = F.binary_cross_entropy(real_scores, 0.9 * ones_target)

            zdis_loss = real_loss + fake_loss
            Dloss += zdis_loss.item()

            zdis_opt.zero_grad()
            zdis_loss.backward()
            zdis_opt.step()

            # Compute AAE loss = reconstruction loss + generator loss for maximizing fooling of zdis
            fake_scores = zdis(fake_codes)
            gen_loss = F.binary_cross_entropy(fake_scores, ones_target)
            Gloss += gen_loss.item()

            reconstructions = autoencoder.decode(fake_codes)
            recon_loss = F.mse_loss(reconstructions, input_features)
            AEloss += recon_loss.item()

            ae_loss = gen_loss + recon_loss

            ae_opt.zero_grad()
            ae_loss.backward()
            ae_opt.step()

        num_batch = len(mnist_train_loader)
        Dloss /= num_batch
        Gloss /= num_batch
        AEloss /= num_batch

        with torch.no_grad():
            prior = torch.rand(len(test_batch), 128).to(device)  # Directly from random uniform distribution [0, 1)
            testX = autoencoder.decode(prior)

            test_features = test_batch.view(-1, 784).to(device)
            testZ = autoencoder.encode(test_features)
            meanZ, stdZ = testZ.mean(), testZ.std()

        print("epoch : {}/{}, D-loss = {:.6f}, G-loss = {:.6f}, AE-loss = {:.6f}, mean-Z = {:.3f}, std-Z = {:.3f}".format(epoch + 1, epochs, Dloss, Gloss, AEloss, meanZ, stdZ))

        save_images(f"AAE-grid-{epoch+1}", testX)

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    torch.save(autoencoder.state_dict(), f"./models/aae{epochs}.pt")
    torch.save(zdis.state_dict(), f"./models/pzdis{epochs}.pt")



if __name__ == "__main__":
    train_adversarial_autoencoder(100)