import torch
import torch.nn as nn

D = 64  # model size coef
LATENT_DIM = 100  # size of the random noise input in the generator


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * D),
            nn.ReLU(),
            nn.Unflatten(1, (16 * D, 4, 4)),
            nn.Upsample(scale_factor=2),  # 8
            nn.Conv2d(16 * D, 8 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(8 * D, 4 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(4 * D, 2 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(2 * D, D, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),  # 64
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 128
            nn.Conv2d(D, D // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 256
            nn.Conv2d(D // 2, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, D, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(D, D * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(D * 2, D * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(D * 4, D * 8, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(D * 8, D * 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256, 256, 1),
        )

    def forward(self, x):
        return self.model(x)
