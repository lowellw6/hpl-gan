import torch

import os
from tqdm import tqdm
import numpy as np
import cv2

from hpl_gan.config import DATASET_PATH, MODEL_PATH, RESULT_PATH
from hpl_gan.ffhq_ALAE.util import save_images, save_reconstructions
from hpl_gan.ffhq_ALAE.ffhq import get_ffhq_test_data

from hpl_gan.ffhq_ALAE.alae.model import Model
from hpl_gan.ffhq_ALAE.alae.tracker import *

# Where to save bulk generations
OUTPATH = "/mnt/slow_ssd/lowell/styleALAE/gen_model_194"

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


def generate(alae_sd, num_batch, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ALAE (adapted from make_recon_figure_ffhq_real.py in original repo)
    alae = load_alae(alae_sd, device)

    for b in tqdm(range(num_batch)):
        # Generate style samples (copied from make_generation_figure.py in ALAE repo)
        # rnd = np.random.RandomState(5)
        styles = np.random.randn(batch_size, 512)
        styles = torch.tensor(styles).float().to(device)

        # Synthesize images with StyleALAE decoder
        generations = alae.generate(8, 1, styles, 1, mixing=True)
        generations = (generations + 1) * 0.5
        
        # save grid to disk
        # name = "styleALAEGen3"
        # save_images(os.path.join(RESULT_PATH, name), generations)

        # save single images to disk
        results = generations.to("cpu")
        results = results.permute(0, 2, 3, 1)
        results = results.detach().numpy()
        results *= 255.
        for i, img in enumerate(results):
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            idx = i + b * batch_size
            name = str(idx).zfill(5)
            path = os.path.join(OUTPATH, name+".png")
            cv2.imwrite(path, out_img)


def reconstruct(alae_sd, num_reconstructions):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ALAE (adapted from make_recon_figure_ffhq_real.py in original repo)
    alae = load_alae(alae_sd, device)

    # Helpers for encoding and decoding with ALAE
    def encode(x):
        Z, _ = alae.encode(x, LAYER_COUNT - 1, 1)
        Z = Z.repeat(1, alae.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        return alae.decoder(x, LAYER_COUNT - 1, 1, noise=True)

    val_path = os.path.join(FULL_FFHQ_DATA_PATH, "val")
    ffhq_test_loader = get_ffhq_test_data(store_location=val_path, num_workers=4)
    test_batch, _ = next(iter(ffhq_test_loader)) 

    w = encode((test_batch.to(device) * 2) - 1)
    x = decode(w)
    x = (x + 1) * 0.5

    save_reconstructions(os.path.join(RESULT_PATH, "styleALAERec"), test_batch[:num_reconstructions], x[:num_reconstructions])


if __name__ == "__main__":
    #generate("ALAE/model_194.pth", 50, 64)
    reconstruct("ALAE/model_194.pth", 4)