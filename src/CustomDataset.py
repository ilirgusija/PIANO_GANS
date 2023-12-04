from torch.utils.data import Dataset
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []

        for subdir, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(".npy"):
                    path = os.path.join(subdir, file)
                    self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_data = np.load(img_path)

        # Add a channel dimension if missing
        img_data = np.expand_dims(
            img_data, axis=0
        )  # Assuming it's a single channel image

        return img_data
