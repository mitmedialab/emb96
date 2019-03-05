from dataset.midi import MEASURES, NOTES, SAMPLE_PER_MEASURE
from model.network import Decoder
from dataset.midi import img2midi
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def generate(output_dir, latent_size, momentum, checkpoint, n):
    decoder = Decoder((MEASURES, NOTES, SAMPLE_PER_MEASURE), latent_size, momentum)
    check   = torch.load(checkpoint)
    decoder.load_state_dict(check['decoder_state_dict'])
    decoder = decoder.cuda()
    decoder.eval()

    z = torch.randn((n, latent_size)).cuda()
    y = decoder(z)

    y = y.detach().cpu().numpy()
    y = [np.hstack(y[i]) for i in range(n)]

    fig  = plt.figure(figsize=(10, 8))
    axes = [fig.add_subplot(n, 1, i + 1) for i in range(n)]

    for i in tqdm(range(n), total=n, desc='Ploting generated pianorolls'):
        axes[i].clear()
        axes[i].imshow(y[i])
        axes[i].axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    plt.show()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(n), total=n, desc='Generating midi files'):
        img2midi(y[i], os.path.join(output_dir, f'{i+1:04d}.mid'))
