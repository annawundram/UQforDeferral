import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py


class AIROGS_ood(Dataset):
    def __init__(self, file_path, transform=None):
        hf = h5py.File(file_path, "r")

        self.transform = transform
        self.levels = hf["level"]
        self.images = hf["images"]
        self.labels = hf["diagnosis"]

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size) 
        image = np.moveaxis(image, -1, 0)

        # Convert to torch tensor
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            label = torch.tensor(label, dtype=torch.long, device="cuda")
        else: 
            label = torch.tensor(label, dtype=torch.long, device="cpu")
        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.images.shape[0]