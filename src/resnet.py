import torch
import torch.nn as nn
import torchvision.models as models

LATENT_DIM=100

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # First, transform the input 1D noise vector into a 2D shape
        self.fc = nn.Linear(LATENT_DIM, self.init_size * 2 * 2)
        self.resnet = models.resnet18(pretrained=True)

    def forward(self, X):
        return self.resnet(X)
