import torch

import os
from tqdm import tqdm
import numpy as np
import cv2

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq_ALAE.model import LatentGeneratorStyleALAE
from hpl_gan.ffhq_ALAE.util import save_images, save_reconstructions
from hpl_gan.ffhq_ALAE.ffhq import get_ffhq_test_data

from hpl_gan.ffhq_ALAE.alae.model import Model
from hpl_gan.ffhq_ALAE.alae.tracker import *


# Where to save bulk generations
OUTPATH = "/mnt/slow_ssd/lowell/styleALAE/LAN_280_ALAE_194"

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


def generate(alae_sd, zgen_sd, num_batch, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ALAE (adapted from make_recon_figure_ffhq_real.py in original repo)
    alae = load_alae(alae_sd, device)

    def decode(x):
        return alae.decoder(x, LAYER_COUNT - 1, 1, noise=True)

    # Load pre-trained LAN generator
    zgen = LatentGeneratorStyleALAE(1024).to(device)
    zgen.load_state_dict(torch.load(os.path.join(MODEL_PATH, zgen_sd), map_location=device))
    zgen.eval()

    for b in tqdm(range(num_batch)):
        # Generate style samples using learned LAN mapping
        prior = (torch.rand(batch_size, 1024).to(device) * 2) - 1
        styles = zgen(prior)
        
        # Synthesize images with StyleALAE decoder
        generations = (decode(styles) + 1) * 0.5
        
        # save grid to disk
        name = f"styleLANGen{b+1}"
        save_images(os.path.join(RESULT_PATH, name), generations)

        # save single images to disk
        # results = generations.to("cpu")
        # results = results.permute(0, 2, 3, 1)
        # results = results.detach().numpy()
        # results *= 255.
        # for i, img in enumerate(results):
        #     out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     idx = i + b * batch_size
        #     name = str(idx).zfill(5)
        #     path = os.path.join(OUTPATH, name+".png")
        #     cv2.imwrite(path, out_img)


if __name__ == "__main__":
    generate("ALAE/model_194.pth", "ffhqZGen280.pt", 3, 4)