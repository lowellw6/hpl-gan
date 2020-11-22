import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm

from model import AE, LatentDiscriminatorMLP, LatentGeneratorMLP
from mnist import get_mnist_train_data, get_mnist_test_data
from eval import save_images


def train_hpl(epochs, state_dict_path_AE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_train_loader = get_mnist_train_data(store_location="./datasets")
    
    mnist_test_loader = get_mnist_test_data(store_location="./datasets")
    test_batch, _ = next(iter(mnist_test_loader))  # Used for visual checkpoints of progress

    autoencoder = AE(input_shape=784).to(device)
    if state_dict_path_AE:
        autoencoder.load_state_dict(torch.load(state_dict_path_AE, map_location=device))

    zgen = LatentGeneratorMLP(128, 128).to(device)
    zgen_opt = optim.Adam(zgen.parameters(), lr=1e-3)

    zdis = LatentDiscriminatorMLP(128).to(device)
    zdis_opt = optim.Adam(zdis.parameters(), lr=1e-3)

    with torch.no_grad():
        testZ = zgen(torch.rand(len(test_batch), 128).to(device))
        testX = autoencoder.decode(testZ)
    save_images(f"HPL-grid-0", testX, device)
    
    print(f"Training HPL transfer mapping for {epochs} epochs on MNIST digits")
    for epoch in range(epochs):
        Dloss = 0
        Gloss = 0
        for image_tensors, _ in tqdm(mnist_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            input_features = image_tensors.view(-1, 784).to(device)
            real_codes = autoencoder.encode(input_features)
            real_codes = real_codes.detach()

            real_scores = zdis(real_codes)
            real_loss = F.binary_cross_entropy(real_scores, ones_target)

            prior = torch.rand(len(image_tensors), 128).to(device)
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy(fake_scores, zeros_target)

            zdis_loss = real_loss + fake_loss
            Dloss += zdis_loss.item()

            zdis_opt.zero_grad()
            zdis_loss.backward()
            zdis_opt.step()

            # Compute generator loss for maximizing fooling of zdis
            prior = torch.rand(len(image_tensors), 128).to(device)
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes)
            zgen_loss = F.binary_cross_entropy(fake_scores, ones_target)
            Gloss += zgen_loss.item()

            zgen_opt.zero_grad()
            zgen_loss.backward()
            zgen_opt.step()

        Dloss /= len(mnist_train_loader)
        Gloss /= len(mnist_train_loader)

        with torch.no_grad():
            testZ = zgen(torch.rand(len(test_batch), 128).to(device))
            testX = autoencoder.decode(testZ)
            meanZ, stdZ = testZ.mean(), testZ.std()

        print("epoch : {}/{}, D-loss = {:.6f}, G-loss = {:.6f}, mean-Z = {:.3f}, std-Z = {:.3f}".format(epoch + 1, epochs, Dloss, Gloss, meanZ, stdZ))

        save_images(f"HPL-grid-{epoch+1}", testX, device)

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    torch.save(zgen.state_dict(), f"./models/zgen{epochs}.pt")
    torch.save(zdis.state_dict(), f"./models/zdis{epochs}.pt")



if __name__ == "__main__":
    train_hpl(100, "./models/ae20.pt")