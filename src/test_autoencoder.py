import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from model import Generator, ResnetGenerator

# Set your hyperparameters and device
LATENT_DIM = 100

# Setup for DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs.")

# params_names = ['autoencoder_model_10', 'autoencoder_model_20', 'autoencoder_model_11', 'autoencoder_model_30', 'autoencoder_model']
params_names = ['autoencoder_model']
for params_name in params_names:
    generator = Generator().to(device)
    state_dict = torch.load(f"../params/resnet_gen/{params_name}.pth", map_location=device)
    # Load the trained model
    using_data_parallel = False
    # Wrap models with DataParallel if more than one GPU is available
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        using_data_parallel = True


    # Adjust the keys based on whether you are using DataParallel or not
    if using_data_parallel:  # Set this based on your model
        # Add 'module.' prefix to each key
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    else:
        # Remove 'module.' prefix
        new_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}

        # Load the adjusted state dictionary into the model
        generator.load_state_dict(new_state_dict)
    generator.to(device)
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
    plt.savefig(f"../output/{params_name}_resnet.png")
