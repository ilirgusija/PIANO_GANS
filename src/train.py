import datetime
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

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
EPOCH = 5000
D = 64  # model size coef
LATENT_DIM = 100  # size of the random noise input in the generator
IMG_DIM = 256, 256, 1  # size of input images and produced images

STFT_ARRAY_DIR = "../data/resized_stft/"
AUDIO_OUT_DIR = "../data/images/"


def wasserstein_loss(y_true, y_pred):
    """for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    return torch.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    gradients = torch.autograd.grad(
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


def get_dataset_paths(directory, extension):
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(subdir, file)
                paths.append(path)
    return paths


# load parameters for audio reconstruction
with open(STFT_ARRAY_DIR + "my_cache.json") as f:
    cache = json.load(f)
print(cache)

# load training data
paths = get_dataset_paths(STFT_ARRAY_DIR, ".npy")
X_train_ = np.zeros(shape=(len(paths), IMG_DIM[0], IMG_DIM[1]))
for i, path in enumerate(paths):
    X_train_[i, :, :] = np.load(paths[i])
X_train_ = X_train_[:, :, :, None]

# Now we initialize the generator and discriminator.
generator = Generator()
discriminator = Discriminator()

training_set_size = X_train_.shape[0]
indices = np.arange(training_set_size)
number_batches = int(training_set_size / BATCH_SIZE)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(EPOCH):
    np.random.shuffle(indices)
    print("Epoch: ", epoch)
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(number_batches):
        discriminator.train()
        generator.train()
        discriminator_minibatches = X_train_[
            indices[i * minibatches_size : (i + 1) * minibatches_size]
        ]

        # Training Discriminator
        for j in range(TRAINING_RATIO):
            disc_optimizer.zero_grad()
            real_samples = discriminator_minibatches[
                j * BATCH_SIZE : (j + 1) * BATCH_SIZE
            ]
            real_samples = torch.tensor(
                real_samples, device=device, dtype=torch.float32
            )

            noise = torch.rand(BATCH_SIZE, LATENT_DIM, device=device)
            generated_samples = generator(
                noise
            ).detach()  # Detach to avoid backprop through G

            gp_loss = gradient_penalty_loss(
                discriminator, real_samples, generated_samples, device
            )

            disc_real = discriminator(real_samples)
            disc_fake = discriminator(generated_samples)

            disc_loss = -torch.mean(disc_real) + torch.mean(disc_fake) + gp_loss
            disc_loss.backward()
            disc_optimizer.step()

        # Training Generator
        gen_optimizer.zero_grad()
        noise = torch.rand(BATCH_SIZE, LATENT_DIM, device=device)
        generated_samples = generator(noise)
        gen_loss = -torch.mean(discriminator(generated_samples))
        gen_loss.backward()
        gen_optimizer.step()

        # Generate images and save audio per epoch
        # Adapt generate_images and save_audio functions to work with PyTorch tensors and operations
        generate_images(generator, "./output/", epoch, cache)

        # Save models at the end of each epoch
        torch.save(
            generator.state_dict(),
            os.path.join(AUDIO_OUT_DIR, f"generator_epoch_{epoch}.pth"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(AUDIO_OUT_DIR, f"discriminator_epoch_{epoch}.pth"),
        )
