import torch.nn as nn
import torch

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.contiguous().view(x.size(0), *self.size)

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        x       = x.contiguous().view(x.size(0), x.size(1), -1)
        b, t, s = x.size()
        x       = x.contiguous().view(b * t, s)
        y       = self.module(x)
        y       = y.contiguous().view(b, t, -1)

        return y

class Encoder(nn.Module):
    def __init__(self, size, beta, latent_size):
        super(Encoder, self).__init__()
        self.size        = size
        self.beta        = beta
        self.latent_size = latent_size
        t, h, w          = size

        self.frame_encoder = TimeDistributed(nn.Sequential(
            nn.Linear(h * w, 2000), nn.ReLU(inplace=True),
            nn.Linear( 2000,  200), nn.ReLU(inplace=True)
        ))

        self.flatten       = Flatten()

        self.song_encoder  = nn.Sequential(
            nn.Linear(t * 200, 1600), nn.ReLU(inplace=True)
        )
        self.mu            = nn.Linear(1600, self.latent_size)
        self.sigma         = nn.Linear(1600, self.latent_size)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).normal_(mean=0., std=self.beta).cuda()

        return mu + std * eps

    def forward(self, x):
        y      = self.frame_encoder(x)
        y      = self.flatten(y)
        y      = self.song_encoder(y)
        mu     = self.mu(y)
        logvar = self.sigma(y)
        z      = self.reparametrization(mu, logvar)

        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, size, latent_size, momentum):
        super(Decoder, self).__init__()
        self.size        = size
        self.latent_size = latent_size
        t, h, w          = size

        self.song_decoder  = nn.Sequential(
            nn.Linear(self.latent_size,    1600), nn.BatchNorm1d(   1600, momentum=momentum), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(            1600, t * 200), nn.BatchNorm1d(t * 200, momentum=momentum), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )

        self.unflatten     = UnFlatten((t, 200))

        self.frame_decoder = TimeDistributed(nn.Sequential(
            nn.Linear( 200,  2000), nn.BatchNorm1d( 2000, momentum=momentum), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(2000, h * w),                                           nn.Sigmoid()
        ))
        self.final         = UnFlatten((t, h, w))

    def forward(self, x):
        y = self.song_decoder(x)
        y = self.unflatten(y)
        y = self.frame_decoder(y)
        y = self.final(y)

        return y
