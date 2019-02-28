import torch.nn as nn
import torch

class FrameEncoder(nn.Module):
    def __init__(self):
        super(FrameEncoder, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x_reshaped    = x.contiguous().view(b * t, c, h, w)
        y             = self.convs(x_reshaped)
        return y.contiguous().view(b, t * y.size(1))

class SongEncoder(nn.Module):
    def __init__(self):
        super(SongEncoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fe = FrameEncoder()
        self.se = SongEncoder()

    def forward(self, x):
        return self.se(self.fe(x))

class FrameDecoder(nn.Module):
    def __init__(self, t, c, h, w):
        super(FrameDecoder, self).__init__()
        self.t = t
        self.c = c
        self.h = h
        self.w = w

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=0, output_padding=0),
        )

    def forward(self, x):
        b          = x.size(0)
        x_reshaped = x.contiguous().view(b * self.t, 128, 1, 1)
        y          = self.convs(x_reshaped)
        return y.contiguous().view(b, self.t, self.c, self.h, self.w)

class SongDecoder(nn.Module):
    def __init__(self):
        super(SongDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, t, c, h, w):
        super(Decoder, self).__init__()

        self.sd = SongDecoder()
        self.fd = FrameDecoder(t, c, h, w)

    def forward(self, x):
        return self.fd(self.sd(x))


# if __name__ == '__main__':
#     import numpy as np
#
#     encoder = Encoder()
#     decoder = Decoder(16, 1, 96, 96)
#     print(decoder(encoder(
#         torch.from_numpy(np.zeros((1, 16, 1, 96, 96), dtype=np.uint8)).float()
#     )).size())
