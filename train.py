import librosa
import numpy as np
from PIL import Image


def load_and_split_audio(file_path, segment_length=10):
    audio, sr = librosa.load(file_path, sr=None)
    segment_samples = sr * segment_length
    segments = [
        audio[i : i + segment_samples] for i in range(0, len(audio), segment_samples)
    ]
    return segments, sr


def audio_to_spectrogram(audio_segment, sr, target_shape=(256, 256)):
    spectrogram = np.abs(librosa.stft(audio_segment))
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram_image = Image.fromarray(log_spectrogram, "RGB").resize(target_shape)
    return spectrogram_image


file_path = ""
segments, sample_rate = load_and_split_audio(file_path)

for segment in segments:
    spectrogram = audio_to_spectrogram(segment, sample_rate)
    # convert the image to a matrix and normalize
    spectrogram_matrix = np.array(spectrogram)
    spectrogram_matrix = spectrogram_matrix / 255.0  # input to generator
