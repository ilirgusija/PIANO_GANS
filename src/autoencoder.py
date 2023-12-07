import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, ResnetGenerator
from CustomDataset import CustomDataset
import matplotlib.pyplot as plt
import numpy as np

# Set your hyperparameters
LATENT_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5  # For early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs.")

generator = ResnetGenerator()
if num_gpus > 1:
    generator = nn.DataParallel(generator)
generator.to(device)

# DC_GAN_DIR = "../data/dc_gan_stuff/"
STFT_ARRAY_DIR = "../data/resized_stft/"
data_set = CustomDataset(data_dir=STFT_ARRAY_DIR)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

# Using AdamW optimizer with weight decay for regularization
optimizer = optim.AdamW(generator.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

criterion = nn.MSELoss()
epoch_loss = []
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        images = data.to(device)
        current_batch_size = images.size(0)
        noise = np.random.rand(current_batch_size, LATENT_DIM).astype(np.float32)
        noise = torch.from_numpy(noise).to(device)
        optimizer.zero_grad()
        outputs = generator(noise)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss_avg = running_loss / len(data_loader)
    epoch_loss.append(epoch_loss_avg)
    scheduler.step(epoch_loss_avg)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss_avg}")

    # Early Stopping
    if epoch_loss_avg < best_loss:
        best_loss = epoch_loss_avg
        epochs_no_improve = 0
        torch.save(generator.state_dict(), f"../params/resnet_gen/autoencoder_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            print("Early stopping triggered")
            break

# Plotting the loss graph
plt.figure(figsize=(12, 7))
plt.plot(epoch_loss, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("autoencoder_loss.png")
