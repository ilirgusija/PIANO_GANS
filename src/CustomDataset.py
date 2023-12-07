from torch.utils.data import Dataset
import numpy as np
import os


from torch.utils.data import Dataset
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, replace_nan_with=None):
        self.data_files = [] 

        for filename in os.listdir(data_dir):
            if not filename.endswith('.npy'):
                continue

            file_path = os.path.join(data_dir, filename)
            try:
                data = np.load(file_path, allow_pickle=True)
                if np.isnan(data).any():
                    if replace_nan_with is not None:
                        print(f"NaN values found in {filename}. Replacing with {replace_nan_with}.")
                        data = np.nan_to_num(data, nan=replace_nan_with)
                    else:
                        print(f"NaN values found in {filename}. Skipping this file.")
                        continue
                self.data_files.append(data) 
            except ValueError as e:
                print(f"Error loading {filename}: {e}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_data = self.data_files[idx]  # Retrieve the data directly

        # Perform the data checks
        self.check_data_for_issues(img_data)
        self.check_data_range(img_data)

        # Add a channel dimension if missing
        if img_data.ndim == 2:  # Checking number of dimensions
            img_data = np.expand_dims(img_data, axis=0)

        return img_data

    @staticmethod
    def check_data_for_issues(data):
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")

    @staticmethod
    def check_data_range(data, expected_min=-1, expected_max=1):
        if np.min(data) < expected_min or np.max(data) > expected_max:
            raise ValueError(f"Data range out of bounds: {np.min(data)}, {np.max(data)}")