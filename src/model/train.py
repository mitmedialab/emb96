from dataset.midi import NOTES, SAMPLE_PER_MEASURE, MEASURES
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

def plot(fig, axes, _imgs, imgs):
    _imgs = np.hstack(_imgs.reshape(MEASURES, NOTES, SAMPLE_PER_MEASURE))
    imgs  = np.hstack(imgs.reshape(MEASURES, NOTES, SAMPLE_PER_MEASURE))

    axes[0].clear()
    axes[0].set_title('output')
    axes[0].imshow(_imgs)
    axes[0].axis('off')

    axes[1].clear()
    axes[1].set_title('output > 0.5')
    axes[1].imshow((_imgs > 0.5 * _imgs.max()).astype(np.uint8))
    axes[1].axis('off')

    axes[2].clear()
    axes[2].set_title('output > 0.75')
    axes[2].imshow((_imgs > 0.75 * _imgs.max()).astype(np.uint8))
    axes[2].axis('off')

    axes[3].clear()
    axes[3].set_title('target')
    axes[3].imshow(imgs)
    axes[3].axis('off')

    plt.tight_layout()
    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)

    data = PIL.Image.open(buf)
    data = ToTensor()(data)

    return data

def train(epochs, batch_size, learning_rate, weight_decay,
          beta_1, beta_2, latent_size, momentum, num_workers, dataset_dir,
          experience_name, saving_rate, checkpoint=None, cpu=False):

    experience_name = os.path.join('../', experience_name)
    if not os.path.isdir(experience_name):
        os.mkdir(experience_name)

    model_dir = os.path.join(experience_name, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    log_dir = os.path.join(experience_name, 'log')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    writer     = SummaryWriter(log_dir=log_dir)
    dataset    = Dataset(dataset_dir)
    params     = { 'num_workers': num_workers, 'batch_size': batch_size }
    dataloader = data.DataLoader(dataset, shuffle=True, **params)

    encoder = Encoder((MEASURES, NOTES, SAMPLE_PER_MEASURE), beta_1, latent_size)
    decoder = Decoder((MEASURES, NOTES, SAMPLE_PER_MEASURE), latent_size, momentum)

    start_epoch = 0
    if checkpoint is not None:
        check = torch.load(checkpoint)
        encoder.load_state_dict(check['encoder_state_dict'])
        decoder.load_state_dict(check['decoder_state_dict'])
        start_epoch = int(check['epoch'])

    criterion = Criterion(beta_2)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr           = learning_rate,
        weight_decay = weight_decay
    )

    if not cpu:
        encoder   = encoder.cuda()
        decoder   = decoder.cuda()
        criterion = criterion.cuda()

    fig  = plt.figure(figsize=(20, 10))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]

    for epoch in range(start_epoch, epochs):
        encoder.train()
        decoder.train()

        reduced_loss  = 0.
        reduced_recon = 0.
        reduced_kld   = 0.
        pbar          = tqdm(dataloader, desc=f'Epoch trainval {epoch + 1}/{epochs}')

        for batch_id, (imgs, labels) in enumerate(pbar):
            if not cpu:
                imgs   = imgs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            z, mu, logvar = encoder(imgs)
            _imgs         = decoder(z)

            loss, recon, kld = criterion(_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            reduced_loss  += loss.item()
            reduced_recon += recon.item()
            reduced_kld   += kld.item()
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
                    'reduced_recon'       : reduced_recon,
                    'reduced_kld'         : reduced_kld,
                    'beta_1'              : beta_1,
                    'beta_2'              : beta_2,
                    'latent_size'         : latent_size
                }, os.path.join(model_dir, f'model_{epoch + 1}.pt'))

                writer.add_scalar('reduced_loss',  reduced_loss,  epoch + 1)
                writer.add_scalar('reduced_recon', reduced_recon, epoch + 1)
                writer.add_scalar('reduced_kld',   reduced_kld,   epoch + 1)

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
                    imgs, labels = dataset[i]
                    imgs         = imgs.unsqueeze(0)
                    labels       = labels.unsqueeze(0)

                    if not cpu:
                        imgs   = imgs.cuda()
                        labels = labels.cuda()

                    z, _, _ = encoder(imgs)
                    _imgs   = decoder(z)[0].cpu().detach().numpy()
                    imgs    = imgs[0].cpu().detach().numpy()

                    writer.add_image(
                        f'entry_{i}',
                        plot(fig, axes, _imgs, imgs),
                        epoch + 1
                    )
