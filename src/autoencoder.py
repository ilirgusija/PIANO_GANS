import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator
from CustomDataset import CustomDataset

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, generator):
        super(Autoencoder, self).__init__()
        self.generator = generator

    def forward(self, x):
        return self.generator(x)

# Set your hyperparameters
LATENT_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

# Initialize the generator and autoencoder
generator = Generator()
autoencoder = Autoencoder(generator)

STFT_ARRAY_DIR = "../data/resized_stft/"

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

train_set = CustomDataset(data_dir=STFT_ARRAY_DIR)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Pretrain the autoencoder
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in train_loader:
        # Assuming data is a batch of images, no need for labels in autoencoder training
        images, _ = data
        
        # Flatten the images and generate random noise as input
        images = images.view(images.size(0), -1)
        noise = torch.randn(images.size(0), LATENT_DIM)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = autoencoder(noise)
        
        # Compute the loss
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader)}")

torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')
