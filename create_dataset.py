import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np 
import sys


path_to_search = " C:\Projects\Manga_Illustration_pix2pix\\train"

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.file_list = os.listdir(self.file_path)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_path[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        width = image.shape[2]
        if self.transform:
            image = self.transform(image)

        # Split the image into input and target
        input_image = image[:, :width//2, :]
        target_image = image[:, width//2:, :]
        
        return input_image, target_image

dataset = CustomDataset(path_to_search)
loader = DataLoader(dataset, batch_size=5)
for x, y in loader:
        print(x.shape)
        sys.exit()

