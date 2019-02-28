from torch.nn import functional as F

import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, _y, y, mu, logvar):
        recon = F.binary_cross_entropy(_y, y, reduction='sum')
        return recon
