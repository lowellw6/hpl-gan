import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
from tqdm import tqdm

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq.model import ConvEncoder, ConvGenerator, LatentDiscriminatorMLP, LatentGeneratorMLP
from hpl_gan.ffhq.ffhq import get_ffhq_train_data, get_ffhq_test_data
from hpl_gan.ffhq.util import save_images


def train_hpl(epochs, gen_state_dict, enc_state_dict, chk_gap=None):
    """
    Trains latent-space generator which attempts to map the prior Z distribution to
    match the latent distribution of an AutoEncoder.

    Starts with a pre-trained AutoEncoder (which is frozen and unchanged).
    """
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

    z_size = 1024  # Latent representation vector size

    # Load pixel-space generator and encoder
    xgen = ConvGenerator(z_size=z_size, conv_reshape=(64, 4, 4)).to(device)
    if gen_state_dict:
        xgen.load_state_dict(torch.load(os.path.join(MODEL_PATH, gen_state_dict), map_location=device))

    xenc = ConvEncoder(z_size=z_size).to(device)
    if enc_state_dict:
        xenc.load_state_dict(torch.load(os.path.join(MODEL_PATH, enc_state_dict), map_location=device))

    xgen.eval()
    xenc.eval()

    # Initialize latent-space generator and discriminator
    zgen = LatentGeneratorMLP(z_size, 256).to(device)
    zgen_opt = optim.Adam(zgen.parameters(), lr=2e-4)

    zdis = LatentDiscriminatorMLP(z_size, 64).to(device)
    zdis_opt = optim.Adam(zdis.parameters(), lr=2e-4)

    print(zgen)
    print(zdis)

    # Initialize log file
    with open(os.path.join(RESULT_PATH, "progress.csv"), 'x') as f:
        f.write("Epoch,ZGenTrainLoss,ZGenValLoss,ZDisTrainLoss,ZDisValLoss,MeanHplZ,StdHplZ\n")

    # Initial visualization (autoencoder generator with random vectors from Z-space as input)
    with torch.no_grad():
        prior = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
        post = zgen(prior)
        generations = (xgen(post) + 1) * 0.5
    save_images(os.path.join(RESULT_PATH, "FFHQ-Thumbnails-HPL-grid-0"), generations)
    
    print(f"Training HPL transfer mapping for {epochs} epochs on FFHQ thumbnails")
    for epoch in range(epochs):
        # Train loop
        zgen.train()
        zdis.train()
        zgen_train_loss = 0
        zdis_train_loss = 0
        pause_zgen = False
        pause_zdis = False
        for image_tensors, _ in tqdm(ffhq_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            input_features = ((image_tensors * 2) - 1).to(device)
            real_codes = xenc(input_features).detach()

            real_scores = zdis(real_codes)
            real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)  # Smoothed "real" label

            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

            combined_zdis_loss = real_loss + fake_loss
            zdis_train_loss += combined_zdis_loss.item()

            if not pause_zdis:
                zdis_opt.zero_grad()
                combined_zdis_loss.backward()
                zdis_opt.step()

            # Compute generator loss for maximizing fooling of zdis
            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes)
            hpl_transfer_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)
            zgen_train_loss += hpl_transfer_loss.item()

            if not pause_zgen:
                zgen_opt.zero_grad()
                hpl_transfer_loss.backward()
                zgen_opt.step()

            # Pause either latent generator or latent discriminator if one is winning too much
            win_ratio = combined_zdis_loss.item() / hpl_transfer_loss.item()
            if win_ratio < 0.1:
                pause_zdis = True
            elif win_ratio > 10:
                pause_zgen = True
            else:
                pause_zdis = False
                pause_zgen = False

        zdis_train_loss /= len(ffhq_train_loader)
        zgen_train_loss /= len(ffhq_train_loader)

        # Evaluation loop
        zgen.eval()
        zdis.eval()
        zgen_val_loss = 0
        zdis_val_loss = 0
        mean_hpl_z = 0
        std_hpl_z = 0
        for image_tensors, _ in ffhq_test_loader:
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Discriminator evaluation
            input_features = ((image_tensors * 2) - 1).to(device)
            real_codes = xenc(input_features).detach()

            real_scores = zdis(real_codes)
            real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)  # Smoothed "real" label

            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

            combined_zdis_loss = real_loss + fake_loss
            zdis_val_loss += combined_zdis_loss.item()

            mean_hpl_z += fake_codes.mean().item()
            std_hpl_z += fake_codes.std().item()

            # Generator evaluation
            prior = (torch.rand(len(image_tensors), z_size).to(device) * 2) - 1
            fake_codes = zgen(prior)
            fake_scores = zdis(fake_codes)
            hpl_transfer_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)
            zgen_val_loss += hpl_transfer_loss.item()

            mean_hpl_z += fake_codes.mean().item()
            std_hpl_z += fake_codes.std().item()

        zdis_val_loss /= len(ffhq_test_loader)
        zgen_val_loss /= len(ffhq_test_loader)
        mean_hpl_z /= len(ffhq_test_loader)
        std_hpl_z /= len(ffhq_test_loader)

        # Terminal update
        print("epoch : {}/{}".format(epoch + 1, epochs))
        print("Z-Dis : train-loss = {:.6f}, val-loss = {:.6f}".format(zdis_train_loss, zdis_val_loss))
        print("Z-Gen : train-loss = {:.6f}, val-loss = {:.6f}".format(zgen_train_loss, zgen_val_loss))
        print("Stats : mean-Z = {:.3f}, std-Z = {:.3f}".format(mean_hpl_z, std_hpl_z))

        # Log dump
        with open(os.path.join(RESULT_PATH, "progress.csv"), 'a') as f:
            f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{:.3f}\n".format(
                epoch+1, zgen_train_loss, zgen_val_loss, zdis_train_loss, zdis_val_loss, mean_hpl_z, std_hpl_z
                )
            )

        # Save visualization of progress
        with torch.no_grad():
            prior = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
            post = zgen(prior)
            generations = (xgen(post) + 1) * 0.5
        save_images(os.path.join(RESULT_PATH, f"FFHQ-Thumbnails-HPL-grid-{epoch+1}"), generations)

        # Take model checkpoint if at appropriate epoch
        if chk_gap and (epoch + 1) % chk_gap == 0:
            torch.save(zgen.state_dict(), os.path.join(MODEL_PATH, f"ffhqZGen{epoch+1}.pt"))
            torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epoch+1}.pt"))

    # Always save parameters after done training
    torch.save(zgen.state_dict(), os.path.join(MODEL_PATH, f"ffhqZGen{epoch+1}.pt"))
    torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epoch+1}.pt"))


if __name__ == "__main__":
    train_hpl(1000, "convAE_1000/convGen1000.pt", "convAE_1000/convEnc1000.pt", 20)