import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
from tqdm import tqdm
import numpy as np

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq_ALAE.model import LatentDiscriminatorStyleALAE, LatentGeneratorStyleALAE
from hpl_gan.ffhq_ALAE.ffhq import get_ffhq_train_data, get_ffhq_test_data
from hpl_gan.ffhq_ALAE.util import save_images, save_reconstructions

from hpl_gan.ffhq_ALAE.alae.model import Model
from hpl_gan.ffhq_ALAE.alae.tracker import *

# full FFHQ dataset stored on seperate SSD drive due to size
FULL_FFHQ_DATA_PATH = "/mnt/slow_ssd/lowell/ffhq/images1024x1024"


LAYER_COUNT = 9

def load_alae(alae_sd, device):
    alae = Model(
        startf=16,
        layer_count=LAYER_COUNT,
        maxf=512,
        latent_size=512,
        truncation_psi=0.7,
        truncation_cutoff=8,
        mapping_layers=8,
        channels=3,
        generator="GeneratorDefault",
        encoder="EncoderDefault"
    ).to(device)
    alae.eval()
    alae.requires_grad_(False)

    decoder = alae.decoder
    encoder = alae.encoder
    mapping_tl = alae.mapping_tl
    mapping_fl = alae.mapping_fl
    dlatent_avg = alae.dlatent_avg

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    trained_weights = torch.load(os.path.join(MODEL_PATH, alae_sd), map_location=device)
    for name, model in model_dict.items():
        if name in trained_weights["models"]:
            trained_md = trained_weights["models"].pop(name)
            model_dict[name].load_state_dict(trained_md, strict=False)

    return alae


def train_hpl_on_ALAE(epochs, alae_sd, chk_gap=None):
    """
    Trains latent-space generator which attempts to map the prior Z distribution to
    match the latent distribution of an AutoEncoder.

    Starts with a pre-trained ALAE (which is frozen and unchanged).
    https://arxiv.org/pdf/2004.04467.pdf
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    train_path = os.path.join(FULL_FFHQ_DATA_PATH, "train")
    val_path = os.path.join(FULL_FFHQ_DATA_PATH, "val")

    ffhq_train_loader = get_ffhq_train_data(store_location=train_path, num_workers=4)
    
    ffhq_test_loader = get_ffhq_test_data(store_location=val_path, num_workers=4)
    test_batch, _ = next(iter(ffhq_test_loader))  # Used for visual checkpoints of progress

    # Load pre-trained ALAE (adapted from make_recon_figure_ffhq_real.py in original repo)
    alae = load_alae(alae_sd, device)
    
    # Helpers for encoding and decoding with ALAE
    def encode(x):
        Z, _ = alae.encode(x, LAYER_COUNT - 1, 1)
        Z = Z.repeat(1, alae.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        return alae.decoder(x, LAYER_COUNT - 1, 1, noise=True)

    # w = encode((test_batch.to(device) * 2) - 1)
    # x = decode(w)

    # t = decode(torch.rand(16, 18, 512).to(device))

    # print(w.shape)
    # print(x.shape)
    # print(F.mse_loss((test_batch.to(device) * 2) - 1, x))
    # print(F.mse_loss((test_batch.to(device) * 2) - 1, t))

    # x = (x + 1) * 0.5
    # print(x.min(), x.max())

    # save_reconstructions(os.path.join(RESULT_PATH, "test"), test_batch[:2], x[:2])


    z_size = 1024

    # Initialize latent-space generator and discriminator
    zgen = LatentGeneratorStyleALAE(z_size).to(device)
    zgen_opt = optim.Adam(zgen.parameters(), lr=2e-4)

    zdis = LatentDiscriminatorStyleALAE(z_size).to(device)
    zdis_opt = optim.Adam(zdis.parameters(), lr=2e-4)

    # prior = (torch.rand(16, z_size).to(device) * 2) - 1
    # styles = zgen(prior)
    # score = zdis(styles)

    print(zgen)
    print(zdis)

    # Initialize log file
    with open(os.path.join(RESULT_PATH, "progress.csv"), 'x') as f:
        f.write("Epoch,ZGenTrainLoss,ZGenValLoss,ZDisTrainLoss,ZDisValLoss,MeanHplZ,StdHplZ\n")

    # Initial visualization (autoencoder generator with random vectors from Z-space as input)
    with torch.no_grad():
        prior = (torch.rand(4, z_size).to(device) * 2) - 1
        styles = zgen(prior)
        generations = (decode(styles) + 1) * 0.5
    save_images(os.path.join(RESULT_PATH, "FFHQ-HPL-grid-0"), generations)
    
    print(f"Training HPL transfer mapping for {epochs} epochs on FFHQ")
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch + 1, epochs))
        # Train loop
        zgen.train()
        zdis.train()
        zgen_train_loss = 0
        zdis_train_loss = 0
        pause_zgen = False
        pause_zdis = False
        print("Training...")
        for image_tensors, _ in tqdm(ffhq_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            input_features = ((image_tensors * 2) - 1).to(device)
            real_codes = encode(input_features).detach()

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
        print("Evaluating...")
        for image_tensors, _ in tqdm(ffhq_test_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Discriminator evaluation
            input_features = ((image_tensors * 2) - 1).to(device)
            real_codes = encode(input_features).detach()

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
        print("Saving examples generations...")
        with torch.no_grad():
            prior = (torch.rand(4, z_size).to(device) * 2) - 1
            styles = zgen(prior)
            generations = (decode(styles) + 1) * 0.5
        save_images(os.path.join(RESULT_PATH, f"FFHQ-HPL-grid-{epoch+1}"), generations)

        # Take model checkpoint if at appropriate epoch
        if chk_gap and (epoch + 1) % chk_gap == 0:
            print("Saving model checkpoints...")
            torch.save(zgen.state_dict(), os.path.join(MODEL_PATH, f"ffhqZGen{epoch+1}.pt"))
            torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epoch+1}.pt"))

    # Always save parameters after done training
    torch.save(zgen.state_dict(), os.path.join(MODEL_PATH, f"ffhqZGen{epoch+1}.pt"))
    torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epoch+1}.pt"))



if __name__ == "__main__":
    train_hpl_on_ALAE(300, "ALAE/model_194.pth", 10)