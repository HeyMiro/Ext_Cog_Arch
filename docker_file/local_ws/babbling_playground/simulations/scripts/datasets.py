import os
import torch
# import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BabblingDataset( Dataset ):
    def __init__( self, states_file, root_dir, transform = None ):
        self.states = torch.load(root_dir + states_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__( self ):
        return len(self.states)
    
    def __getitem__( self, idx ):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + "view_" + str(idx) + ".png"
        image = io.imread(img_name)
        state = self.states[idx]
        

        if self.transform:
            image = self.transform( image )

        return image, state

        
def build_dataset( transform ):
    # Builds the dataset to train the neural network
    babbling = BabblingDataset('states.pt', '/home/aljiro/dev/dataset/', transform)

    return babbling
