import torch.nn as nn
import torch


D = 64
LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(LATENT_DIM, 512 * D)
        self.unflatten = nn.Unflatten(1, (32 * D, 4, 4))
        self.relu = nn.LeakyReLU()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(32 * D, 16 * D, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16 * D, 8 * D, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8 * D, 4 * D, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(4 * D, 2 * D, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(2 * D, D, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(D, D // 2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(D // 2, 1, kernel_size=5, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv7(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(1, D, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(D, D * 2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(D * 2, D * 4, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(D * 4, D * 8, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(D * 8, D * 16, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(D * 16 * 16 * 16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
