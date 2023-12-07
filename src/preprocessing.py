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
AUDIO_OUT_DIR = "../output"
STFT_ARRAY_DIR = "../data/stft_arrays"
DC_GAN_DIR = "../data/dc_gan_stuff"
PROCESSED_STFT_DIR = "../data/clipped_stft"
RESIZED_STFT_DIR = "../data/resized_stft"

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

def display_spectrogram(directory, save=False):
    """
    Function used to generate and display sample spectrogram from audio files.
    """
    paths = prep_utils.get_absolute_file_paths(directory)
    count = 0
    start_time = time.time()
    i = 0

    while count < 3 and i < len(paths):  # Ensure to not go beyond the list length
        print(f"count: {count}, i: {i}")
        prep_utils.display_progress_eta(current_item=paths[i], total_items=paths, start_time=start_time)
        
        S = np.load(paths[i])
        print("S Shape: ", S.shape)

        melspec_log = librosa.feature.melspectrogram(S=np.log(S + 1e-7), sr=22050)
        print("MelSpec Shape: ", melspec_log.shape)
        
        # Check for empty or invalid values in melspec_log
        if melspec_log.size == 0 or np.isnan(melspec_log).any() or np.isinf(melspec_log).any():
            print("Warning: MelSpec is empty or contains NaNs/Infinities. Skipping specshow.")
            i += 1  # Move to the next file
            continue

        plt.figure()
        librosa.display.specshow(melspec_log, y_axis='mel', x_axis='time')
        plt.colorbar()
        if save:
            plt.savefig(f"../output/spectrogram_{os.path.split(directory)[-1]}.png")
        else:
            plt.show()
        
        count += 1
        i += 1  # Move to the next file

def display_a_spectrogram(S):
    """
    Function used to generate and display sample spectrogram from audio files.
    """
    print("S Shape: ", S.shape)

    melspec_log = librosa.feature.melspectrogram(S=np.log(S+1e-7), sr=22050)
    print("MelSpec Shape: ", melspec_log.shape)
    
    # Check for empty or invalid values in melspec_log
    if melspec_log.size == 0 or np.isnan(melspec_log).any() or np.isinf(melspec_log).any():
        print("Warning: MelSpec is empty or contains NaNs/Infinities. Skipping specshow.")
        return

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
    start_time = time.time()
    index = 0
    for path in paths:
        # prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        S = np.load(path)
        # Initial Spectrogram
        # display_a_spectrogram(S)

        # Logarithm and NaN handling
        S_log = np.log1p(S)  # Safer logarithm
        S_log = np.nan_to_num(S_log, nan=-999, posinf=-999, neginf=-999)  # Replace NaNs and Infinities

        # Normalization
        S_normalized = (S_log - means[index]) / (stds[index] + 1e-6)

        # Clipping
        S_clipped = np.where(np.abs(S_normalized) < 3, S_normalized, 3 * np.sign(S_normalized))

        # Rescaling
        S_rescaled = S_clipped / 3

        # Save the processed array
        out = PROCESSED_STFT_DIR + prep_utils.get_filename(path) + ".npy"
        np.save(out, S_rescaled)
        index+=1

def downsample():
    """
    Downsample and resize to 512x512, and save them locally
    """
    paths = prep_utils.get_absolute_file_paths(PROCESSED_STFT_DIR)
    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)
        S = np.load(path)
        S_downsample = skimage.transform.resize(S, (512, 512), anti_aliasing=True)
        out = RESIZED_STFT_DIR + prep_utils.get_filename(path) + ".npy"
        np.save(out, S_downsample)

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

def preprocessing():
    """
    Data processing for SpecGAN
    """
    make_audio_chunks(seconds=20, dest_dir=AUDIO_CHUNKS_20S_DIR)
    # display_spectrogram()
    convert_audio_to_stft(src_dir=AUDIO_CHUNKS_20S_DIR, dest_dir=STFT_ARRAY_DIR, extension=".npy")
    # audio_reconstruction()
    record_mean_std()
    preprocessing_arrays()
    downsample()

def dc_gan_processing():
    paths = prep_utils.get_dataset_paths(AUDIO_CHUNKS_20S_DIR, ".wav")

    start_time = time.time()
    for path in paths:
        prep_utils.display_progress_eta(current_item=path, total_items=paths, start_time=start_time)

        y, sr = librosa.load(path, sr=22050)

        # Acquire magnitude matrix
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S, phase = librosa.magphase(D) 

        # normalize S and downsample
        S = normalize_stft(S)
        S = cv2.resize(S, (512, 512), interpolation=cv2.INTER_AREA)

        _, file_name = os.path.split(path)
        out = DC_GAN_DIR + os.path.splitext(file_name)[0] + ".npy"

        np.save(out, S)

def check_invalid_files_in_dataset(directory):
    """
    Parses the given directory for .npy files and checks each file for NaN values.
    Returns the count of files containing NaN values.

    :param directory: The directory containing the dataset files.
    :return: Count of invalid files containing NaN values.
    """
    invalid_files_count = 0
    invalid_files = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                data = np.load(file_path)
                # Check if the file contains NaN values
                if np.isnan(data).any():
                    invalid_files_count += 1
                    invalid_files.append(filename)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    
    # Optionally, print out the names of invalid files
    print("Invalid files containing NaN values:")
    for file in invalid_files:
        print(file)
    print(f"Total invalid files: {invalid_files_count}")
    return invalid_files_count

if __name__ == "__main__":
    preprocessing()
    create_cache_file(RESIZED_STFT_DIR, 512, 512, 512)
    # dc_gan_processing()
    # display_spectrogram(STFT_ARRAY_DIR, True)
    # display_spectrogram(PROCESSED_STFT_DIR, True)
    # display_spectrogram(RESIZED_STFT_DIR, True)

