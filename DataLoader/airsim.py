import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


ow,oh = 384,216
class Airsim(data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.img_dir = img_dir


    def __getitem__(self, idx):
        image_name = self.img_dir + self.frame.ix[idx, 0]
        label_name = self.img_dir + self.frame.ix[idx, 1]

        img = Image.open(image_name).convert('RGB')
        img = img.resize((ow,oh),Image.BILINEAR)
        target = Image.open(label_name)


        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target, image_name, label_name


    def __len__(self):
        return len(self.frame)



class Airsim_img(data.Dataset):

    def __init__(self, img_dir, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.img_dir = img_dir


    def __getitem__(self, idx):
        image_name = self.img_dir + self.frame.ix[idx, 0]
        label_name = self.img_dir + self.frame.ix[idx, 1]

        img = Image.open(image_name).convert('RGB')
        img = img.resize((ow,oh),Image.BILINEAR)
        target = Image.open(label_name)


        if self.transform:
            img, _ = self.transform(img, target)
            
        return img, image_name


    def __len__(self):
        return len(self.frame)        