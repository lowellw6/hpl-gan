import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
from tqdm import tqdm

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq.model import ConvGenerator, ConvEncoder, LatentDiscriminatorMLP
from hpl_gan.ffhq.ffhq import get_ffhq_train_data, get_ffhq_test_data
from hpl_gan.ffhq.util import save_reconstructions, save_images


def train_adversarial_autoencoder(epochs, chk_gap=None):
    """
    Note "real_codes" and "fake_codes" are reversed compared to HPL
    real --> from prior ~ Uniform [0, 1)
    fake --> from AAE encoder (generator)
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
    test_input = ((test_batch * 2) - 1).to(device)

    z_size = 1024  # Latent representation vector size

    gen = ConvGenerator(z_size=z_size, conv_reshape=(64, 4, 4)).to(device)

    enc = ConvEncoder(z_size=z_size).to(device)  # Starts with pre-trained alexnet for convolutional features

    zdis = LatentDiscriminatorMLP(z_size, 64).to(device)

    print(gen)
    print(enc)
    print(zdis)
    
    ae_params = list(gen.parameters()) + list(enc.parameters())
    ae_opt = optim.Adam(ae_params, lr=2e-4)

    zdis_opt = optim.Adam(zdis.parameters(), lr=2e-4)

    # Initialize log file
    with open(os.path.join(RESULT_PATH, "progress.csv"), 'x') as f:
        f.write("Epoch,AETrainLoss,AEValLoss,EncFoolTrainLoss,EncFoolValLoss,ZDisTrainLoss,ZDisValLoss,MeanAAEZ,StdAAEZ\n")

    # Initial visualization (AAE generator with random vectors from Z-space as input)
    with torch.no_grad():
        prior = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
        generations = (gen(prior) + 1) * 0.5
    save_images(os.path.join(RESULT_PATH, "FFHQ-Thumbnails-AAE-synth-0"), generations)

    # Initial reconstructions (AAE encoding then generation from and to pixel-space)
    with torch.no_grad():
        z = enc(test_input)
        x = gen(z)
        reconstructions = (x + 1) * 0.5
    save_reconstructions(os.path.join(RESULT_PATH, "FFHQ-Thumbnails-AAE-recon-0"), test_batch, reconstructions)

    print(f"Training AAE for {epochs} epochs on FFHQ thumbnails")
    for epoch in range(epochs):
        # Train loop
        gen.train()
        enc.train()
        zdis.train()
        recon_train_loss = 0
        enc_fool_train_loss = 0
        zdis_train_loss = 0
        pause_enc_fool = False
        pause_zdis = False
        
        for image_tensors, _ in tqdm(ffhq_train_loader):
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Compute discriminator loss with real and fake latent codes
            input_features = ((image_tensors * 2) - 1).to(device)
            fake_codes = enc(input_features)

            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

            real_codes = torch.rand(len(image_tensors), z_size).to(device)
            real_scores = zdis(real_codes)
            real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)

            combined_zdis_loss = fake_loss + real_loss
            zdis_train_loss += combined_zdis_loss.item()

            if not pause_zdis:
                zdis_opt.zero_grad()
                combined_zdis_loss.backward()
                zdis_opt.step()

            # Compute AAE loss = reconstruction loss + encoder loss for maximizing fooling of zdis
            fake_scores = zdis(fake_codes)
            enc_fool_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)  
            enc_fool_loss_val = enc_fool_loss.item()     
            enc_fool_train_loss += enc_fool_loss_val

            if pause_enc_fool:
                enc_fool_loss *= 0  # supresses gradients from encoder fool loss

            reconstructions = gen(fake_codes)
            recon_loss = F.mse_loss(reconstructions, input_features)
            recon_train_loss += recon_loss.item()

            ae_loss = enc_fool_loss + recon_loss

            ae_opt.zero_grad()
            ae_loss.backward()
            ae_opt.step()

            # Pause either encoder fool or latent discriminator loss if one is winning too much
            win_ratio = combined_zdis_loss.item() / enc_fool_loss_val
            if win_ratio < 0.1:
                pause_zdis = True
            elif win_ratio > 10:
                pause_enc_fool = True
            else:
                pause_zdis = False
                pause_enc_fool = False

        num_batch = len(ffhq_train_loader)
        recon_train_loss /= num_batch
        enc_fool_train_loss /= num_batch
        zdis_train_loss /= num_batch

        # Evaluation loop
        gen.eval()
        enc.eval()
        zdis.eval()
        recon_val_loss = 0
        enc_fool_val_loss = 0
        zdis_val_loss = 0
        mean_aae_z = 0
        std_aae_z = 0
        
        for image_tensors, _ in ffhq_test_loader:
            ones_target = torch.ones(len(image_tensors), 1).to(device)
            zeros_target = torch.zeros(len(image_tensors), 1).to(device)

            # Discriminator evaluation
            input_features = ((image_tensors * 2) - 1).to(device)
            fake_codes = enc(input_features)

            fake_scores = zdis(fake_codes.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_scores, zeros_target)

            real_codes = torch.rand(len(image_tensors), z_size).to(device)
            real_scores = zdis(real_codes)
            real_loss = F.binary_cross_entropy_with_logits(real_scores, 0.9 * ones_target)

            combined_zdis_loss = fake_loss + real_loss
            zdis_val_loss += combined_zdis_loss.item()

            mean_aae_z += fake_codes.mean().item()
            std_aae_z += fake_codes.std().item()

            # AAE evaluation
            fake_scores = zdis(fake_codes)
            enc_fool_loss = F.binary_cross_entropy_with_logits(fake_scores, ones_target)  
            enc_fool_val_loss += enc_fool_loss.item()

            reconstructions = gen(fake_codes)
            recon_loss = F.mse_loss(reconstructions, input_features)
            recon_val_loss += recon_loss.item()

        num_batch = len(ffhq_test_loader)
        recon_val_loss /= num_batch
        enc_fool_val_loss /= num_batch
        zdis_val_loss /= num_batch
        mean_aae_z /= num_batch
        std_aae_z /= num_batch

        # Terminal update
        print("epoch    : {}/{}".format(epoch + 1, epochs))
        print("Recon    : train-loss = {:.6f}, val-loss = {:.6f}".format(recon_train_loss, recon_val_loss))
        print("Enc-Fool : train-loss = {:.6f}, val-loss = {:.6f}".format(enc_fool_train_loss, enc_fool_val_loss))
        print("Z-Dis    : train-loss = {:.6f}, val-loss = {:.6f}".format(zdis_train_loss, zdis_val_loss))
        print("Stats    : mean-Z = {:.3f}, std-Z = {:.3f}".format(mean_aae_z, std_aae_z))

        # Log dump
        with open(os.path.join(RESULT_PATH, "progress.csv"), 'a') as f:
            f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{:.3f}\n".format(
                epoch+1, recon_train_loss, recon_val_loss, enc_fool_train_loss, enc_fool_val_loss, zdis_train_loss, zdis_val_loss, mean_aae_z, std_aae_z
                )
            )

        # Save visualization of synthesis progress
        with torch.no_grad():
            prior = (torch.rand(len(test_batch), z_size).to(device) * 2) - 1
            generations = (gen(prior) + 1) * 0.5
        save_images(os.path.join(RESULT_PATH, f"FFHQ-Thumbnails-AAE-synth-{epoch+1}"), generations)    

        # Save visualization of reconstruction progress
        with torch.no_grad():
            z = enc(test_input)
            x = gen(z)
            reconstructions = (x + 1) * 0.5
        save_reconstructions(os.path.join(RESULT_PATH, f"FFHQ-Thumbnails-AAE-recon-{epoch+1}"), test_batch, reconstructions)

        # Take model checkpoint if at appropriate epoch
        if chk_gap and (epoch + 1) % chk_gap == 0:
            torch.save(gen.state_dict(), os.path.join(MODEL_PATH, f"ffhqGen{epoch+1}.pt"))
            torch.save(enc.state_dict(), os.path.join(MODEL_PATH, f"ffhqEnc{epoch+1}.pt"))
            torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epoch+1}.pt"))

    # Always save parameters after done training
    torch.save(gen.state_dict(), os.path.join(MODEL_PATH, f"ffhqGen{epochs}.pt"))
    torch.save(enc.state_dict(), os.path.join(MODEL_PATH, f"ffhqEnc{epochs}.pt"))
    torch.save(zdis.state_dict(), os.path.join(MODEL_PATH, f"ffhqZDis{epochs}.pt"))



if __name__ == "__main__":
    train_adversarial_autoencoder(1000, 100)
