import numpy as np
import matplotlib.pyplot as plt

# Load a sample .npy file
data = np.load('../data/resized_stft/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav_chunk_0.npy')

# Check shape
print("Data Shape:", data.shape)

# Check range
print("Min Value:", np.min(data))
print("Max Value:", np.max(data))

# Visualize the data (if it's an image)
if data.shape == (512, 512) or data.shape == (1, 512, 512):
    plt.imshow(data.squeeze(), cmap='gray')  # Use 'squeeze' to handle single-channel data
    plt.show()
else:
    print("Data shape is not as expected.")