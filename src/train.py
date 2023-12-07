import datetime
import librosa
import numpy as np
from PIL import Image
import torch
import os
import skimage.transform
import pandas as pd
from model import Generator, Discriminator, ResnetDiscriminator, ResnetGenerator
import json
import scipy.io.wavfile
import torch.optim as optim
import torch.nn as nn
from CustomDataset import CustomDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

DC_GAN_DIR = "../data/dc_gan_stuff/"
STFT_ARRAY_DIR = "../data/resized_stft/"
AUDIO_OUT_DIR = "../data/images/"
PARAM_DIR="../params/"
OUTPUT_DIR="../output/"
D = 64
LATENT_DIM = 100

def compute_gradient_penalty(model, real_samples, fake_samples, gradient_penalty_weight=10):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (
        alpha * real_samples + ((1 - alpha) * fake_samples)
    ).requires_grad_(True)
    d_interpolates = model(interpolates)
    fake = torch.ones(
        d_interpolates.size(), requires_grad=False, device=real_samples.device
    )

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*gradient_penalty_weight
    return gradient_penalty

def wasserstein_loss(y_true, y_pred):
    """for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    return torch.mean(y_true * y_pred)

def tile_images(image_stack):
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

def generate_images(generator_model, output_dir, epoch, cache, device):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    noise = torch.randn(5, LATENT_DIM).to(device)  # Generate random noise
    with torch.no_grad():  # Ensure no gradients are computed
        test_image_stack = generator_model(noise)  # Generate images

    # generate and save sample audio file for each epoch
    for i in range(5):
        w = test_image_stack[i]
        outfile = os.path.join(output_dir, "train_epoch_%02d(%02d).wav" % (epoch, i))
        save_audio(w, outfile, cache)

    test_image_stack = (test_image_stack * 255.5) + 255.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode="L")  # L specifies greyscale
    outfile = os.path.join(output_dir, "epoch_{}.png".format(epoch))
    tiled_output.save(outfile)

def save_audio(y, path, cache):
    """generate a wav file from a given spectrogram and save it"""
    s = np.squeeze(y)
    s = skimage.transform.resize(s, (cache["input_freq"], cache["input_time"]))
    s = s * 3
    s = s * (cache["Std Magnitude"] + cache["eps"]) + cache["Mean Magnitude"]
    s = np.exp(s)
    y = librosa.griffinlim(s, hop_length=int(cache["hop_length"]))
    scipy.io.wavfile.write(path, cache["sampling_rate"], y)

def plot_and_save_loss_graph(disc_loss_log, gen_loss_log, epoch, output_dir):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(disc_loss_log, label="Discriminator")
    plt.plot(gen_loss_log, label="Generator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_epoch_{epoch}.png"))
    plt.close()

def train(n_epochs, generator, discriminator, batch_size, training_ratio, gen_optimizer, disc_optimizer, data_loader, cache, device):
    disc_loss_log = []
    gen_loss_log = []
    
    for epoch in range(1, n_epochs + 1):
        print("Epoch: ", epoch)
        for i, real_samples in enumerate(data_loader):
            real_samples = real_samples.to(device)
            discriminator.train()
            generator.train()
            
            # Training Discriminator
            for j in range(training_ratio):
                disc_optimizer.zero_grad()
                # Generate fake data from the generator
                current_batch_size = real_samples.size(0)
                noise = np.random.rand(current_batch_size, LATENT_DIM).astype(np.float32)
                noise = torch.from_numpy(noise).to(device)

                fake_samples = generator(noise).detach()
                # print(real_samples.shape)

                disc_real_output = discriminator(real_samples)
                disc_fake_output = discriminator(fake_samples)
                disc_loss_wasserstein = wasserstein_loss(disc_fake_output, disc_real_output)

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples)

                # Total discriminator loss
                disc_loss = disc_loss_wasserstein + gradient_penalty
                disc_loss.backward()
                disc_optimizer.step()

            # Training Generator
            gen_optimizer.zero_grad()
            noise = torch.rand(batch_size, LATENT_DIM, device=device)
            generated_samples = generator(noise)
            gen_loss = -torch.mean(discriminator(generated_samples))
            gen_loss.backward()
            gen_optimizer.step()
            disc_loss_log.append(disc_loss.item())
            gen_loss_log.append(gen_loss.item())

            print(f"Epoch [{epoch}/{n_epochs}], Batch Step [{i}/{len(data_loader)}], "f"Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")
            # if i % 100 == 0:
            #     print(f"Epoch [{epoch}/{n_epochs}], Batch Step [{i}/{len(data_loader)}], "f"Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")
        # Save models at the end of each epoch
        torch.save(
            generator.state_dict(),
            os.path.join(PARAM_DIR, f"generator_epoch_{epoch}.pth"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(PARAM_DIR, f"discriminator_epoch_{epoch}.pth"),
        )

        plot_and_save_loss_graph(disc_loss_log, gen_loss_log, epoch, OUTPUT_DIR)
        generate_images(generator, AUDIO_OUT_DIR, epoch, cache, device)

def main(epochs, batch_size, training_ratio = 5, using_pretrained=False):
    # load parameters for audio reconstruction
    with open(STFT_ARRAY_DIR + "my_cache.json") as f:
        cache = json.load(f)
        print("Cache loaded!")
        
    # Setup for DataParallel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs.")
    
    # Initialize models
    generator = ResnetGenerator()
    discriminator = ResnetDiscriminator()

    using_data_parallel = False
    # Wrap models with DataParallel if more than one GPU is available
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        using_data_parallel = True
    
    if using_pretrained:
        # Load the state dictionary from the file
        state_dict = torch.load('../params/resnet_gen/autoencoder_model.pth')

        # Adjust the keys based on whether you are using DataParallel or not
        if using_data_parallel:
            # Add 'module.' prefix to each key
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            # Remove 'module.' prefix
            new_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}

        # Load the adjusted state dictionary into the model
        generator.load_state_dict(new_state_dict)
    generator.to(device)
    discriminator.to(device)

    train_set = CustomDataset(data_dir=STFT_ARRAY_DIR)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

    # Training loop
    train(epochs, generator, discriminator, batch_size, training_ratio, gen_optimizer, disc_optimizer, train_loader, cache, device)

if __name__ == "__main__":
    main(100, 64)
