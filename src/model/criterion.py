from torch.nn import functional as F

import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, beta):
        super(Criterion, self).__init__()
        self.beta = beta

    def forward(self, _y, y, mu, logvar):
        recon = F.binary_cross_entropy(_y, y, reduction='sum')
        kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kld
