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

STFT_ARRAY_DIR = "./data/stft_arrays/"
AUDIO_OUT_DIR = "./data/images/"


def wasserstein_loss(y_true, y_pred):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    return torch.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
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


class RandomWeightedAverage(tf.keras.layers.Layer):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""

    def call(self, inputs):  # _merge_function
        weights =torch.random.uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


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
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)


def save_audio(y, path, cache):
    """ generate a wav file from a given spectrogram and save it """
    s = np.squeeze(y)
    s = skimage.transform.resize(s, (cache['input_freq'], cache['input_time']))
    s = s * 3
    s = s * (cache['Std Magnitude'] + cache['eps']) + cache['Mean Magnitude']
    s = np.exp(s)
    y = librosa.griffinlim(s, hop_length=int(cache['hop_length']))
    scipy.io.wavfile.write(path, cache['sampling_rate'], y)


def swap_input(model, input_layer):
    """ swap model input/output.  TODO: Are there any smarter way? """
    x = input_layer
    for layer in model.layers[1:-1]:
        x = layer(x)
    real_fake = model.layers[-1](x)
    return real_fake


def write_log(log_path, names_g, logs_g, names_d, logs_d, batch_no, epoch):
    for name, value in zip(names_d, logs_d):
        losses[name].append(value)
    losses[names_g].append(logs_g)
    losses["epoch"].append(epoch)
    losses["nb_batch_g"].append(batch_no)
    df = pd.DataFrame(losses, columns=losses.keys())
    df.to_json(log_path + "loss.json")


def get_dataset_paths(directory, extension):
    paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(subdir, file)
                paths.append(path)
    return paths

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# load parameters for audio reconstruction
with open(STFT_ARRAY_DIR + 'my_cache.json') as f:
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


# EVERYTHING HERE DOWN NEEDS TO CHANGE

#########################################################################

#########################################################################

#########################################################################

#########################################################################

#########################################################################

### Generator ### 
# for layer in discriminator.layers:
#     layer.trainable = False
# discriminator.trainable = False
# generator.trainable = True

# generator_input = Input(shape=(LATENT_DIM,))  # LATENT_DIM=100 = dimension of random input vector
# generator_layers = generator(generator_input)

# # replace input layer of discriminator with generator output
# d_layers_for_generator = swap_input(discriminator, generator_layers)
# generator_model = Model(inputs=[generator_input], outputs=[d_layers_for_generator])
# generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss])
# generator_model.summary()

# for layer in discriminator.layers:
#     layer.trainable = True
# for layer in generator.layers:
#     layer.trainable = False
# discriminator.trainable = True
# generator.trainable = False

# real_samples = Input(shape=X_train_.shape[1:])
# generator_input_for_discriminator = Input(shape=(LATENT_DIM,))  # random seed input
# generated_samples_for_discriminator = generator(generator_input_for_discriminator)  # random seed -> generator
# d_output_from_generator = swap_input(discriminator, generated_samples_for_discriminator) # # random seed -> generator -> discriminator
# d_output_from_real_samples = swap_input(discriminator, real_samples) # real spectrogram images -> discriminator_loss_real

# # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
# averaged_samples = RandomWeightedAverage()(inputs=[real_samples, generated_samples_for_discriminator])
# averaged_samples_out = swap_input(discriminator, averaged_samples) # weighted-averages of real and generated samples -> discriminator

# # The gradient penalty loss function: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
# partial_gp_loss = partial(gradient_penalty_loss,
#                           averaged_samples=averaged_samples,
#                           gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
# partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# # input: real samples / random seed for generator
# # output: real samples -> discriminator / random seed -> generator -> discriminator_loss /
# #         weighted-averages of real and generated samples -> discriminator / real samples -> discriminator -> categorical output
# discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
#                             outputs=[d_output_from_real_samples, d_output_from_generator,
#                                      averaged_samples_out])
# # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# # samples and the gradient penalty loss for the averaged samples
# discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
#                             loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
# discriminator_model.summary()


##############################################################################################################
# Training
# labels
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)  # for real samples
negative_y = -positive_y                                # for generated fake samples
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)  # passed to the gradient_penalty loss function and is not used.

# Store losses for each training step
now = datetime.datetime.now()
datestr = now.strftime("%Y-%m-%d_%H%M%S")
log_path = './logs/'
g_names = "generator_loss_ws"
d_names = ["discriminator_loss_real", "discriminator_loss_fake", "discriminator_loss_averaged"]
losses = {"epoch": [], "nb_batch_g": [], "discriminator_loss_real": [],
          "discriminator_loss_fake": [], "discriminator_loss_averaged": [],
          "generator_loss_ws": []}


training_set_size = X_train_.shape[0]
indices = np.arange(training_set_size)
number_batches = int(training_set_size / BATCH_SIZE)

generator.load_weights("./data/images/generator_epoch_33_-0.778.h5")
generate_images(generator, "./output/", 33, cache)

for epoch in range(34, EPOCH):
    np.random.shuffle(indices)

    print("Epoch: ", epoch)
    print("Number of batches: ", number_batches)

    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    batch_per_epoch = int(training_set_size // (BATCH_SIZE * TRAINING_RATIO))
    for i in range(batch_per_epoch):
        discriminator_minibatches = X_train_[indices[i * minibatches_size:(i + 1) * minibatches_size]]

        # training D. D will be trained (TRAINING_RATIO) times more than G
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

            noise = np.random.rand(BATCH_SIZE, LATENT_DIM).astype(np.float32)
            d_logs = discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y])
            nb_batch = (j + i * TRAINING_RATIO) + epoch * (batch_per_epoch * TRAINING_RATIO)

        # training G
        g_logs = generator_model.train_on_batch(np.random.rand(BATCH_SIZE, LATENT_DIM), [positive_y])
        nb_batch = epoch * (batch_per_epoch * TRAINING_RATIO) + i * TRAINING_RATIO

        # write log for each batch training step
        write_log(log_path, g_names, g_logs, d_names, d_logs, nb_batch, epoch)

    # export generated images and save sample audio per each epoch
    generate_images(generator, "./output/", epoch, cache)

    # save models each epoch
    outfile = os.path.join(AUDIO_OUT_DIR, 'generator_epoch_{}_{:.3}.h5'.format(epoch, g_logs))
    generator.save_weights(outfile)
    outfile = os.path.join(AUDIO_OUT_DIR, 'discriminator_epoch_{}_{:.3}.h5'.format(epoch, d_logs[0]))
    discriminator.save_weights(outfile)

