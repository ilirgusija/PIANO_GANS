import os
import time

import cv2
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile
import skimage.io
import skimage.transform
from pydub import AudioSegment
from pydub.utils import make_chunks
import json

import preprocessing_utils as prep_utils

DATASET_DIR = "../data/audio/maestro-v3.0.0"
AUDIO_CHUNKS_10S_DIR = "../data/audio/audio_chunks_10s/"
AUDIO_CHUNKS_20S_DIR = "../data/audio/audio_chunks_20s/"
SPECTROGRAM_DIR = "../data/spectrograms/"
AUDIO_OUT_DIR = "../output/"
STFT_ARRAY_DIR = "../data/stft_arrays/"
PROCESSED_STFT_DIR = "../data/clipped_stft/"
RESIZED_STFT_DIR = "../data/resized_stft/"

def make_audio_chunks(seconds, dest_dir):
    """
    Function used to convert audio into shorter audio clips, and save audio clips to files.

    :param seconds: desired clip length
    :param dest_dir: output directory
    """
    paths = prep_utils.get_absolute_file_paths(DATASET_DIR, ".wav")

    start_time = time.time()
    for audio_path in paths:
        prep_utils.display_progress_eta(current_item=audio_path, total_items=paths, start_time=start_time)

        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = seconds * 1000  # 20 seconds
        chunks = make_chunks(audio, chunk_length_ms)
        chunks.pop(-1)

        # Export all of the individual chunks as wav files
        for i, chunk in enumerate(chunks):
            _, chunk_name = os.path.split(os.path.splitext(audio_path)[0] + "_chunk_{0}.wav".format(i))
            chunk.export(dest_dir + chunk_name, format="wav")

    print("\n\nChunks export completed.")

def display_spectrogram():
    """
    Function used to generate and display sample spectrogram from audio files.
    """
    paths = prep_utils.get_absolute_file_paths(AUDIO_CHUNKS_20S_DIR)[:3]

    for path in paths:
        y, sr = librosa.load(path)

        # Decompose a spectrogram with NMF
        # Short-time Fourier transform underlies most analysis.
        # librosa.stft returns a complex matrix D.
        # D[f, t] is the FFT value at frequency f, time (frame) t.
        D = librosa.stft(y)

        # Separate the magnitude and phase and only use magnitude
        S, phase = librosa.magphase(D)
        print("S Shape: ", S.shape)

        melspec_log = librosa.feature.melspectrogram(S=np.log(S), sr=sr)
        print("MelSpec Shape: ", melspec_log.shape)

        plt.figure()
        librosa.display.specshow(melspec_log, y_axis='mel', x_axis='time')
        plt.colorbar()
        plt.show()

def convert_audio_to_stft(src_dir, dest_dir, extension):
    """
    Function used to convert audio clips into Short-Time Fourier Transform matrices, and save matrices to files.

    :param src_dir: input audio directory
    :param dest_dir: output STFT directory
    :param extension: desired output file type
    """
    paths = prep_utils.get_unprocessed_items(src_dir=src_dir, dest_dir=dest_dir)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)

        y, sr = librosa.load(path)

        # Decompose a spectrogram with NMF
        D = librosa.stft(y)

        # Separate the magnitude and phase and only use magnitude
        S, _ = librosa.magphase(D)

        out = dest_dir + prep_utils.get_filename(path) + extension
        np.save(out, S)

# reconstruct arrays into audio clips
def audio_reconstruction():
    """
    Function used to reconstruct sample audio clips from STFT matrices, and save audio to file.
    """
    paths = prep_utils.get_absolute_file_paths(STFT_ARRAY_DIR)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        S = np.load(path)
        y = librosa.griffinlim(S)

        out = AUDIO_OUT_DIR + prep_utils.get_filename(path) + ".wav"

        # Save reconstructed data
        scipy.io.wavfile.write(out, 22050, y)

def create_cache_file(cache_file_path, hop_length, input_freq, input_time, eps=1e-7, sr=22050):
    """
    Creates a cache file containing mean and std of STFT magnitudes, hop length, input frequency, input time, 
    epsilon value, and sampling rate for each audio file.

    :param audio_paths: List of paths to the audio files.
    :param cache_file_path: Path to save the cache file.
    :param hop_length: Hop length used in the STFT.
    :param input_freq: Frequency dimension for resizing the STFT.
    :param input_time: Time dimension for resizing the STFT.
    :param eps: Small constant to prevent division by zero.
    """
    cache_data = {}

    paths = prep_utils.get_absolute_file_paths(STFT_ARRAY_DIR)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        # Compute mean and std of magnitude
        try:
            S = np.load(path)
        except Exception as e:
                print(f"Error loading file {path}: {e}")
                continue
        S = np.log(S)
        mag_mean = float(np.mean(S))
        mag_std = float(np.std(S))
        
        # Save data in cache
        cache_data[path] = {
            'Mean Magnitude': mag_mean,
            'Std Magnitude': mag_std,
            'hop_length': hop_length,
            'input_freq': input_freq,
            'input_time': input_time,
            'eps': eps,
            'sampling_rate': sr
        }

    cache_file_path = os.path.join(cache_file_path, "my_cache.json")
    # Write cache data to JSON file
    with open(cache_file_path, 'w') as fp:
        json.dump(cache_data, fp, indent=4)

    print(f"Cache file created at {cache_file_path}")

def record_mean_std():
    """
    Record mean and std of all STFT matrices and save them locally
    """
    paths = prep_utils.get_absolute_file_paths(STFT_ARRAY_DIR)

    mean_list = []
    std_list = []

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        try:
            S = np.load(path)
        except Exception as e:
                print(f"Error loading file {path}: {e}")
                continue
        S = np.log(S)
        mag_mean = np.mean(S)
        mag_std = np.std(S)
        mean_list.append(mag_mean)
        std_list.append(mag_std)
        print("Finished:", path)

    data = {"mean": mean_list, "std": std_list, "path": paths}
    df = pd.DataFrame.from_dict(data)
    df.to_csv("../data/saved_mean_std.csv")

def preprocessing_arrays():
    """
    Normalize STFT matrices and save them locally
    """
    df = pd.read_csv("../data/saved_mean_std.csv")
    paths = df['path']
    means = df['mean']
    stds = df['std']
    eps = 1e-7

    for index in range(len(paths)):
        print("Processing: ", paths[index])
        S = np.load(paths[index])

        # Processing step for the magnitude matrix of the STFT.
        # Take the logarithm of the magnitudes, normalize it, clip it at 3*std and rescale to [-1,1]
        S = np.log(S)
        S = (S - means[index]) / (stds[index] + eps)
        # clipping
        S = np.where(np.abs(S) < 3, S, 3 * np.sign(S))
        # rescale to [-1,1]
        S /= 3

        out = PROCESSED_STFT_DIR + prep_utils.get_filename(paths[index]) + ".npy"
        np.save(out, S)

def downsample():
    """
    Downsample and resize to 256x256, and save them locally
    """
    paths = prep_utils.get_absolute_file_paths(PROCESSED_STFT_DIR)
    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        S = np.load(path)
        S_downsample = skimage.transform.resize(S, (512, 512), anti_aliasing=True)
        out = RESIZED_STFT_DIR + prep_utils.get_filename(path) + ".npy"
        np.save(out, S_downsample)

def convert_stft_to_images(src_dir, dest_dir, ext=".png", size=None):
    """
    Function used to convert STFT matrices to images, and saves them to destination folder

    :param src_dir: source folder where STFT matrices are stored
    :param dest_dir: output images folder
    :param ext: image format, defaulted to .png
    :param size: dimension of desired square image
    """
    paths = prep_utils.get_unprocessed_items(src_dir=src_dir, dest_dir=dest_dir)

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)

        S_norm = np.load(path)
        S_norm = normalize_stft(S_norm)

        if size:
            S_norm = cv2.resize(S_norm, (size, size), interpolation=cv2.INTER_CUBIC)

        out_path = dest_dir + prep_utils.get_filename(path) + ext
        plt.imsave(out_path, S_norm)

        image = cv2.imread(out_path)
        cv2.imwrite(out_path, image)

def normalize_stft(s):
    """
    Function used to normalize STFT matrix

    :param s: STFT matrix
    """
    s = np.log(s)
    mean = np.mean(s)
    std = np.std(s)
    eps = 1e-7

    s = (s - mean) / (std + eps)

    # clipping
    s = np.where(np.abs(s) < 3, s, 3 * np.sign(s))
    # rescale to [-1,1]
    s /= 3
    return s

def preprocessing():
    """
    Data processing for DCGAN and SpecGAN
    """
    make_audio_chunks(seconds=20, dest_dir=AUDIO_CHUNKS_20S_DIR)
    # display_spectrogram()
    convert_audio_to_stft(src_dir=AUDIO_CHUNKS_20S_DIR, dest_dir=STFT_ARRAY_DIR, extension=".npy")
    # audio_reconstruction()
    record_mean_std()
    preprocessing_arrays()
    downsample()


if __name__ == "__main__":
    # preprocessing()
    # create_cache_file(STFT_ARRAY_DIR, 512, 1025, 862)
    create_cache_file(RESIZED_STFT_DIR, 512, 512, 512)
    # dc_gan_processing()
    # style_gan_preprocessing()
