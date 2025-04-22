import torch
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class ImgDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Dataset class for paired .npy images from a JSON file.

        Args:
            json_path (str): Path to the JSON file.
            transform (callable, optional): Optional transform to be applied on input and/or output.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = list(data.values())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample["input"]
        output_path = sample["output"]

        input_arr = np.load(input_path).astype(np.float32)
        output_arr = np.load(output_path).astype(np.float32)

        # Convert to tensor
        input_tensor = torch.from_numpy(input_arr)
        output_tensor = torch.from_numpy(output_arr)

        
        padding = (0, 672 - output_tensor.size(1), 0, 672 - output_tensor.size(0))  # (left, right, top, bottom)
        output_tensor = F.pad(output_tensor, padding, mode='constant', value=0)  # You can use 'constant' or 'reflect' mode
        padding = (0, 672 - input_tensor.size(1), 0, 672 - input_tensor.size(0))  # (left, right, top, bottom)
        input_tensor = F.pad(input_tensor, padding, mode='constant', value=0)  # You can use 'constant' or 'reflect' mode


        # if self.transform:
            # input_tensor = self.transform(input_tensor)
            # output_tensor = self.transform(output_tensor)
        return input_tensor, output_tensor