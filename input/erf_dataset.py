# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, "%s%s"%(basename,extension))

def image_path_city(root, name):
    return os.path.join(root, "%s"%(name))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class CoIL(Dataset):
    
    def __init__(self, filenames, input_transform=None, target_transform=None):
        self.filenames = filenames
        # self.filenames.sort()

        print("filnames size:", len(self.filenames))

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        #print(filename)

        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
       
        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, 'None', filename, 'None'

    def __len__(self):
        return len(self.filenames)


