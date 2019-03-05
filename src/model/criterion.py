import torch.nn.functional as F
import torch.nn as nn
import torch

class Criterion(nn.Module):
    def __init__(self, beta):
        super(Criterion, self).__init__()
        self.beta = beta
        self.loss = F.binary_cross_entropy

    def forward(self, _y, y, mu, logvar):
        recon = self.loss(_y, y)
        kld   = torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss  = recon - self.beta * kld

        return loss, recon, kld
