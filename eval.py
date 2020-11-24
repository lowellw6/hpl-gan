import torch
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm
import cv2

from model import AE
from mnist import get_mnist_test_data
from util import save_reconstructions



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
    save_reconstructions(f"AE-grid-{epoch+1}", test_batch, reconstructions)






if __name__ == "__main__":
    eval_autoencoder("./models/ae20.pt")