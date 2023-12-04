import torch.nn as nn

BATCH_SIZE = 64
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10
EPOCH = 5000
D = 64
LATENT_DIM = 100
IMG_DIM = (1, 256, 256)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512 * D),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16 * D, 16 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16 * D, 8 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8 * D, 4 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4 * D, 2 * D, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(2 * D, D, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(D, D // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(D // 2, 1, kernel_size=5, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(256, D, kernel_size=5, stride=2, padding=2),
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
            nn.Linear(D * 16 * 16 * 16, 1),
        )

    def forward(self, x):
        return self.model(x)
