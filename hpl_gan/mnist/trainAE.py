import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.mnist.model import AE
from hpl_gan.mnist.mnist import get_mnist_train_data, get_mnist_test_data
from hpl_gan.mnist.util import save_reconstructions


def train_autoencoder(epochs):
    """
    Adapted from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    mnist_train_loader = get_mnist_train_data(store_location=DATASET_PATH)
    
    mnist_test_loader = get_mnist_test_data(store_location=DATASET_PATH)
    test_batch, _ = next(iter(mnist_test_loader))  # Used for visual checkpoints of progress
    test_input = test_batch.view(-1, 784).to(device)
    test_input = (test_input * 2) - 1

    autoencoder = AE(input_shape=784, z_size=128).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    with torch.no_grad():
        reconstructions = autoencoder(test_input)
        reconstructions = (reconstructions + 1) * 0.5
    save_reconstructions(os.path.join(RESULT_PATH, "AE-grid-0"), test_batch, reconstructions)
    
    print(f"Training for {epochs} epochs on MNIST digits")
    for epoch in range(epochs):
        loss = 0
        for image_tensors, _ in tqdm(mnist_train_loader):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            input_features = image_tensors.view(-1, 784).to(device)
            input_features = (input_features * 2) - 1
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            reconstructions = autoencoder(input_features)
            
            # compute training reconstruction loss
            train_loss = F.mse_loss(reconstructions, input_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(mnist_train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

        # save visual results on first batch of held-out set
        with torch.no_grad():
            reconstructions = autoencoder(test_input)
            reconstructions = (reconstructions + 1) * 0.5
        save_reconstructions(os.path.join(RESULT_PATH, f"AE-grid-{epoch+1}"), test_batch, reconstructions)


    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    torch.save(autoencoder.state_dict(), os.path.join(MODEL_PATH, f"ae{epochs}.pt"))



if __name__ == "__main__":
    train_autoencoder(20)