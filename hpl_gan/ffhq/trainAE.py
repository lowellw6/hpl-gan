import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
from tqdm import tqdm

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq.model import ConvGenerator, ConvEncoder
from hpl_gan.ffhq.ffhq import get_ffhq_train_data, get_ffhq_test_data
from hpl_gan.ffhq.util import save_reconstructions


def train_autoencoder(epochs, chk_gap=None):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    
    train_path = os.path.join(DATASET_PATH, "ffhq/thumbnails128x128-train")
    val_path = os.path.join(DATASET_PATH, "ffhq/thumbnails128x128-val")

    ffhq_train_loader = get_ffhq_train_data(store_location=train_path)
    
    ffhq_test_loader = get_ffhq_test_data(store_location=val_path)
    test_batch, _ = next(iter(ffhq_test_loader))  # Used for visual checkpoints of progress
    test_input = ((test_batch * 2) - 1).to(device)

    z_size = 1024  # Latent representation vector size

    gen = ConvGenerator(z_size=z_size, conv_reshape=(64, 4, 4)).to(device)

    enc = ConvEncoder(z_size=z_size).to(device)  # Starts with pre-trained alexnet for convolutional features

    print(gen)
    print(enc)

    params = list(gen.parameters()) + list(enc.parameters())
    optimizer = optim.Adam(params, lr=2e-4)

    # Initialize log file
    with open(os.path.join(RESULT_PATH, "progress.csv"), 'x') as f:
        f.write("Epoch,TrainLoss,ValLoss\n")

    # Initial visualization (will be random noise)
    with torch.no_grad():
        z = enc(test_input)
        x = gen(z)
        reconstructions = (x + 1) * 0.5
    save_reconstructions(os.path.join(RESULT_PATH, "ConvAE-grid-0"), test_batch, reconstructions)
    
    print(f"Training for {epochs} epochs on FFHQ thumbnails")
    for epoch in range(epochs):
        # Train loop
        gen.train()
        enc.train()
        train_loss = 0
        for image_tensors, _ in tqdm(ffhq_train_loader):
            input_features = ((image_tensors * 2) - 1).to(device)
            
            z = enc(input_features)
            recon = gen(z)
            
            recon_loss = F.mse_loss(recon, input_features)
            
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            
            train_loss += recon_loss.item()
        
        train_loss /= len(ffhq_train_loader)

        # Evaluation loop
        gen.eval()
        enc.eval()
        val_loss = 0
        for image_tensors, _ in ffhq_test_loader:
            input_features = ((image_tensors * 2) - 1).to(device)

            z = enc(input_features)
            recon = gen(z)

            recon_loss = F.mse_loss(recon, input_features)
            val_loss += recon_loss.item()

        val_loss /= len(ffhq_test_loader)
        
        # Terminal update
        print("epoch : {}/{}, train-loss = {:.6f}, val-loss = {:.6f}".format(epoch + 1, epochs, train_loss, val_loss))

        # Log dump
        with open(os.path.join(RESULT_PATH, "progress.csv"), 'a') as f:
            f.write("{},{:.6f},{:.6f}\n".format(epoch+1, train_loss, val_loss))

        # Save visualization of progress
        with torch.no_grad():
            z = enc(test_input)
            x = gen(z)
            reconstructions = (x + 1) * 0.5
        save_reconstructions(os.path.join(RESULT_PATH, f"ConvAE-grid-{epoch+1}"), test_batch, reconstructions)

        # Take model checkpoint if at appropriate epoch
        if chk_gap and (epoch + 1) % chk_gap == 0:
            torch.save(gen.state_dict(), os.path.join(MODEL_PATH, f"convGen{epoch+1}.pt"))
            torch.save(enc.state_dict(), os.path.join(MODEL_PATH, f"convEnc{epoch+1}.pt"))

    # Always save parameters after done training
    torch.save(gen.state_dict(), os.path.join(MODEL_PATH, f"convGen{epochs}.pt"))
    torch.save(enc.state_dict(), os.path.join(MODEL_PATH, f"convEnc{epochs}.pt"))



if __name__ == "__main__":
    train_autoencoder(1000, 100)