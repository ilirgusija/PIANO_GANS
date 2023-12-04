import os
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []

        for subdir, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.npy'):
                    path = os.path.join(subdir, file)
                    self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        return os.open(img_path)