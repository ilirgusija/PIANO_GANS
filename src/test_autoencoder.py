import torch
import matplotlib.pyplot as plt
import numpy as np
from model import (
    Generator,
)  # Assuming Generator is the class for your autoencoder model

# Set your hyperparameters and device
LATENT_DIM = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
generator = Generator().to(device)
generator.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
generator.eval()

num_samples = 10
noise = np.random.rand(num_samples, LATENT_DIM).astype(np.float32)
noise = torch.from_numpy(noise).to(device)

# Generate images from random noise
with torch.no_grad():
    generated_images = generator(noise).cpu()

# Display the generated images
plt.figure(figsize=(10, 2))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(np.transpose(generated_images[i], (1, 2, 0)), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()
