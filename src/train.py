import datetime
from CustomDataset import CustomDataset
import librosa
import numpy as np
from PIL import Image
import torch
import os
import skimage.transform
import pandas as pd
from model import Generator, Discriminator
import json
import scipy.io.wavfile
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import grad


BATCH_SIZE = 64
# The training ratio is the number of discriminator updates per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
D = 64  # model size coef
LATENT_DIM = 100  # size of the random noise input in the generator
IMG_DIM = 256, 256, 1  # size of input images and produced images

STFT_ARRAY_DIR = "../data/resized_stft/"
AUDIO_OUT_DIR = "../data/images/"


def wasserstein_loss(y_true, y_pred):
    """for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    return torch.mean(y_true * y_pred)


def gradient_penalty_loss(y_pred, averaged_samples, gradient_penalty_weight):
    """for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    gradients = grad(
        outputs=y_pred,
        inputs=averaged_samples,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_l2_norm = torch.sqrt(torch.sum(torch.square(gradients)))
    gradient_penalty = gradient_penalty_weight * torch.square(1 - gradient_l2_norm)
    return gradient_penalty


def tile_images(image_stack):
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


def generate_images(generator_model, output_dir, epoch, cache):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(5, LATENT_DIM))

    # generate and save sample audio file for each epoch
    for i in range(5):
        w = test_image_stack[i]
        outfile = os.path.join(output_dir, "train_epoch_%02d(%02d).wav" % (epoch, i))
        save_audio(w, outfile, cache)

    test_image_stack = (test_image_stack * 127.5) + 127.5
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


D = 64
LATENT_DIM = 100
IMG_DIM = (1, 256, 256)


def train(
    n_epochs,
    generator,
    discriminator,
    batch_size,
    training_ratio,
    gen_optimizer,
    disc_optimizer,
    data_loader,
    cache,
    device,
):
    for epoch in range(1, n_epochs + 1):
        print("Epoch: ", epoch)
        for i, real_samples in enumerate(data_loader):
            real_samples.to(device)
            discriminator.train()
            generator.train()

            # Training Discriminator
            for j in range(training_ratio):
                disc_optimizer.zero_grad()
                # Generate fake data from the generator
                noise = np.random.rand(batch_size, LATENT_DIM).astype(np.float32)
                noise = torch.from_numpy(noise)

                fake_samples = generator(noise)
                print(real_samples.shape)

                disc_real_output = discriminator(real_samples)
                disc_fake_output = discriminator(fake_samples)
                disc_loss_wasserstein = wasserstein_loss(
                    disc_fake_output, disc_real_output
                )

                # Compute gradient penalty
                gradient_penalty = discriminator.compute_gradient_penalty(
                    real_samples, fake_samples
                )

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

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}], Batch Step [{i}/{len(data_loader)}], "
                    f"Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}"
                )

        if epoch % 25 == 0:
            generate_images(generator, "../output/", epoch, cache)
            # Save models at the end of each epoch
            torch.save(
                generator.state_dict(),
                os.path.join(AUDIO_OUT_DIR, f"generator_epoch_{epoch}.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(AUDIO_OUT_DIR, f"discriminator_epoch_{epoch}.pth"),
            )


def main(b=64):
    # load parameters for audio reconstruction
    with open(STFT_ARRAY_DIR + "my_cache.json") as f:
        cache = json.load(f)
        print("Cache loaded!")
    # Now we initialize the generator and discriminator.
    generator = Generator()
    discriminator = Discriminator()

    train_set = CustomDataset(data_dir=STFT_ARRAY_DIR)
    # train_loader=DataLoader(train_set, batch_size=b, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=b, shuffle=True)

    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

    training_ratio = 5

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        5000,
        generator,
        discriminator,
        b,
        training_ratio,
        gen_optimizer,
        disc_optimizer,
        train_loader,
        cache,
        device,
    )


if __name__ == "__main__":
    main()
