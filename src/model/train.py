from torchvision.transforms import ToTensor
from model.network import Encoder, Decoder
from tensorboardX import SummaryWriter
from dataset.generator import Dataset
from model.criterion import Criterion
from torch.utils import data
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import os
import io

def plot(_imgs, imgs):
    fig  = plt.figure(figsize=(20, 10))
    axes = [fig.add_subplot(16, 2, i + 1) for i in range(32)]

    for i in range(16):
        axes[i].set_title('output')
        axes[i].imshow(_imgs[i])
        axes[i].axis('off')

    for i in range(16):
        axes[16 + i].set_title('target')
        axes[16 + i].imshow(imgs[i])
        axes[16 + i].axis('off')

    plt.tight_layout()
    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)

    data = PIL.Image.open(buf)
    data = ToTensor()(data)

    return data

def train(epochs, batch_size, learning_rate, weight_decay, beta, num_workers,
          dataset_dir, experience_name, saving_rate):

    experience_name = os.path.join('../', experience_name)
    if not os.path.isdir(experience_name):
        os.mkdir(experience_name)

    model_dir = os.path.join(experience_name, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    log_dir = os.path.join(experience_name, 'log')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    dataset  = Dataset(dataset_dir),
    params   = { 'num_workers': num_workers, 'batch_size': batch_size }
    trainval = data.DataLoader(dataset, shuffle=True, **params)


    encoder = Encoder()
    decoder = Decoder(16, 1, 96, 96)

    criterion = Criterion(beta)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay
    )

    encoder   = encoder.cuda()
    decoder   = decoder.cuda()
    criterion = criterion.cuda()

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        reduced_loss = 0.
        pbar         = tqdm(trainval, desc=f'Epoch trainval {epoch + 1}/{epochs}')

        for batch_id, batch in enumerate(pbar):
            imgs, _ = batch
            imgs    = imgs.cuda()

            optimizer.zero_grad()

            z, mu, logvar = encoder(imgs)
            _imgs         = decoder(z)

            loss = criterion(_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            reduced_loss += loss.item()
            pbar.set_postfix(reduced_loss=reduced_loss / (batch_id + 1))

        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            if (epoch + 1) % saving_rate == 0:
                print(f'Saving {epoch + 1}/{epochs}')

                torch.save({
                    'epoch'               : epoch + 1,
                    'encoder_state_dict'  : encoder.state_dict(),
                    'decoder_state_dict'  : decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reduced_loss'        : reduced_loss,
                }, os.path.join(model_dir, f'model_{epoch + 1}.pt'))

                writer.add_scalar('reduced_loss',     reduced_loss,     epoch + 1)

                for name, param in encoder.named_parameters():
                    writer.add_histogram(
                        name,
                        param.clone().cpu().data.numpy(),
                        epoch + 1
                    )

                for name, param in decoder.named_parameters():
                    writer.add_histogram(
                        name,
                        param.clone().cpu().data.numpy(),
                        epoch + 1
                    )

                for i in range(4):
                    batch = Dataset[i]
                    imgs, _ = batch
                    imgs    = imgs.cuda()
                    z, _, _ = encoder(imgs.unsqueeze(0))
                    _imgs   = decoder(z)[0][0].cpu().detach().numpy()
                    writer.add_image(
                        f'entry_{i}',
                        plot(_imgs, imgs[0].cpu().detach().numpy()),
                        epoch + 1
                    )
