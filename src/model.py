import torch.nn as nn
import torch

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
            nn.Linear(LATENT_DIM, 256 * D),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 8
            nn.Conv2d(8 * D, 8 * D, kernel_size=5, padding=2),
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
            nn.Tanh()
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
            nn.Linear(D * 16 * 16 * 16, 1)  # change the input size, idk if its right
        )

    def compute_gradient_penalty(self, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self(interpolates)
        fake = torch.ones(d_interpolates.size(), requires_grad=False, device=real_samples.device)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def forward(self, x):
        return self.model(x)

