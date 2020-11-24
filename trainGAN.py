import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm

from model import AE, PixelDiscriminatorMLP, PixelGeneratorMLP
from mnist import get_mnist_train_data, get_mnist_test_data
from util import save_images


def train_gan(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_train_loader = get_mnist_train_data(store_location="./datasets")
    
    mnist_test_loader = get_mnist_test_data(store_location="./datasets")
    test_batch, _ = next(iter(mnist_test_loader))  # Used for visual checkpoints of progress

    z_size = 16

    gen = PixelGeneratorMLP(z_size, 32, 784).to(device)
    gen_opt = optim.Adam(gen.parameters(), lr=1e-3)

    dis = PixelDiscriminatorMLP(784, 32, 1).to(device)
    dis_opt = optim.Adam(dis.parameters(), lr=1e-3)

    print(gen)
    print(dis)

    with torch.no_grad():
        testZ = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
        testX = (gen(testZ) + 1) * 0.5
    save_images(f"GAN-grid-0", testX, device)
    
    print(f"Training GAN for {epochs} epochs on MNIST digits")
    for epoch in range(epochs):
        Dloss = 0
        Gloss = 0
        for image_tensors, _ in tqdm(mnist_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            real_images = image_tensors.view(-1, 784).to(device)
            real_images = (real_images * 2) - 1

            # real_images += torch.normal(torch.zeros(real_images.shape), torch.ones(real_images.shape) * (1. / (4. * 255.))).to(device)  # add a bit of noise
            # real_images = torch.clamp(real_images, -1, 1)

            real_scores = dis(real_images)
            real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)  # Smoothed "real" label

            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_images = gen(prior)
            fake_scores = dis(fake_images.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

            dis_loss = real_loss + fake_loss
            Dloss += dis_loss.item()

            dis_opt.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            # Compute generator loss for maximizing fooling of dis
            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_images = gen(prior)
            fake_scores = dis(fake_images)
            gen_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)
            Gloss += gen_loss.item()

            # print(gen_loss.item())

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        Dloss /= len(mnist_train_loader)
        Gloss /= len(mnist_train_loader)

        with torch.no_grad():
            testZ = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
            testX = (gen(testZ) + 1) * 0.5
            meanX, stdX = testX.mean(), testX.std()

        print("epoch : {}/{}, D-loss = {:.6f}, G-loss = {:.6f}, mean-X = {:.3f}, std-X = {:.3f}".format(epoch + 1, epochs, Dloss, Gloss, meanX, stdX))

        save_images(f"GAN-grid-{epoch+1}", testX, device)

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    torch.save(gen.state_dict(), f"./models/gen{epochs}.pt")
    torch.save(dis.state_dict(), f"./models/dis{epochs}.pt")



if __name__ == "__main__":
    train_gan(100)