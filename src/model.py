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
        self.batch_norm1 = nn.BatchNorm2d(16 * D)
        self.batch_norm2 = nn.BatchNorm2d(8 * D)
        self.batch_norm3 = nn.BatchNorm2d(4 * D)
        self.batch_norm4 = nn.BatchNorm2d(2 * D)
        self.batch_norm5 = nn.BatchNorm2d(D)
        self.batch_norm6 = nn.BatchNorm2d(D // 2)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.relu(x)
        x = self.upsample1(x)
        
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv7(x)
        x = self.tanh(x)
        return x
    
class ResnetGenerator(nn.Module):
    def __init__(self):
        super(ResnetGenerator, self).__init__()
        self.fc = nn.Linear(LATENT_DIM, 512 * D)
        self.unflatten = nn.Unflatten(1, (32 * D, 4, 4))
        self.relu = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        # Residual blocks
        self.resblock1 = ResidualBlock(32 * D, 16 * D)
        self.resblock2 = ResidualBlock(16 * D, 8 * D)
        self.resblock3 = ResidualBlock(8 * D, 4 * D)
        self.resblock4 = ResidualBlock(4 * D, 2 * D)
        self.resblock5 = ResidualBlock(2 * D, D)
        self.resblock6 = ResidualBlock(D, D // 2)
        self.resblock7 = ResidualBlock(D // 2, 1)
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        x = self.resblock1(x)
        x = self.upsample(x)
        
        x = self.resblock2(x)
        x = self.upsample(x)
        
        x = self.resblock3(x)
        x = self.upsample(x)
        
        x = self.resblock4(x)
        x = self.upsample(x)
        
        x = self.resblock5(x)
        x = self.upsample(x)
        
        x = self.resblock6(x)
        x = self.upsample(x)        

        x = self.resblock7(x)
        x = self.tanh(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride

        # To match the dimensions if in_channels and out_channels are different
        self.match_dimensions = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.match_dimensions(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.leaky_relu(out)

        return out
    
class ResnetDiscriminator(nn.Module):
    def __init__(self):
        super(ResnetDiscriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, D, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.resblock1 = ResidualBlock(D, D * 2, stride=2)
        self.resblock2 = ResidualBlock(D * 2, D * 4, stride=2)
        self.resblock3 = ResidualBlock(D * 4, D * 8, stride=2)
        self.resblock4 = ResidualBlock(D * 8, D * 16, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(D * 16 * 16 * 16, 1)  # Adjust the input size according to your image size

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
# 
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
