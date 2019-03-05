from dataset.midi import MEASURES, NOTES, SAMPLE_PER_MEASURE
from model.network import Decoder
from dataset.midi import img2midi
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
import os
import io

def generate(output_dir, latent_size, momentum, checkpoint, n, cpu=False):
    decoder = Decoder((MEASURES, NOTES, SAMPLE_PER_MEASURE), latent_size, momentum)
    check   = torch.load(checkpoint)
    decoder.load_state_dict(check['decoder_state_dict'])
    if not cpu:
        decoder = decoder.cuda()
    decoder.eval()

    z = torch.randn((n, latent_size))
    if not cpu:
        z = z.cuda()
    y = decoder(z)

    y = y.detach().cpu().numpy()
    y = [np.hstack(y[i]) for i in range(n)]

    fig  = plt.figure(figsize=(10 * 4, n * 4))
    axes = [fig.add_subplot(n, 1, i + 1) for i in range(n)]

    for i in tqdm(range(n), total=n, desc='Ploting generated pianorolls'):
        axes[i].clear()
        axes[i].imshow(y[i])
        axes[i].axis('off')

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0., hspace=0.02)
    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)

    data = PIL.Image.open(buf)
    data.save(os.path.join(output_dir, 'midis.png'))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(n), total=n, desc='Generating midi files'):
        img2midi(y[i], os.path.join(output_dir, f'{i+1:04d}.mid'))
