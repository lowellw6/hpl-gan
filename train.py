import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm

from model import AE
from mnist import get_mnist_train_data, get_mnist_test_data


def train_autoencoder(epochs):
    """
    Adapted from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./datasets"):
        os.mkdir("./datasets")

    mnist_train_loader = get_mnist_train_data(store_location="./datasets")
    mnist_test_loader = get_mnist_test_data(store_location="./datasets")

    autoencoder = AE(input_shape=784).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    print(f"Training for {epochs} epochs on MNIST digits")
    for epoch in range(epochs):
        loss = 0
        for image_tensors, _ in tqdm(mnist_train_loader):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            input_features = image_tensors.view(-1, 784).to(device)
            
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

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    torch.save(autoencoder.state_dict(), f"./models/ae{epochs}.pt")



if __name__ == "__main__":
    train_autoencoder(100)