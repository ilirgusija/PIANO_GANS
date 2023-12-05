import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator
from CustomDataset import CustomDataset
import matplotlib.pyplot as plt
import numpy as np


# Set your hyperparameters
LATENT_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the generator and autoencoder
generator = Generator().to(device)

STFT_ARRAY_DIR = "../data/resized_stft/"

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)


data_set = CustomDataset(data_dir=STFT_ARRAY_DIR)
data_loader = DataLoader(data_set, batch_size=64, shuffle=True)

# Pretrain the autoencoder
epoch_loss = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        images = data
        images = images.to(device)

        # Flatten the images and generate random noise as input
        current_batch_size = images.size(0)
        noise = np.random.rand(current_batch_size, LATENT_DIM).astype(np.float32)
        noise = torch.from_numpy(noise).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = generator(noise)

        # print(outputs.shape)
        # print(images.shape)
        
        # Compute the loss
        loss = criterion(outputs, images)

        # Backward pass and optimization
        loss.backward()
        # print(f"Batch #{i}, loss: {loss.item()}")
        optimizer.step()

        running_loss += loss.item()
    epoch_loss += [running_loss / len(data_loader)]
    print(f"Running loss: {running_loss}, length: {len(data_loader)}")
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(data_loader)}")

torch.save(generator.state_dict(), "autoencoder_model.pth")

plt.figure(figsize=(12, 7))
plt.clf()
plt.plot(epoch_loss, label="train")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc=1)
plt.savefig("autoencoder_loss.png")
