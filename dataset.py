import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CropData(Dataset):
    """
    A custom dataset class for passing the data(images) into Pytorch Dataloader.
    Allows the data passed to be indexed.
    
    Args:
      images: A numpy array of images.
    """

    def __init__(self, images):
        super(CropData, self).__init__()

        self.images = images
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        img = torch.from_numpy(img.transpose((2, 0, 1)).astype("float32")) / 255.
        return img

class LoadImages:
    """
    Loads the images in a given folder.
    
    Args:
      base: the path to the folder containing the images.
      dsize: the new shape to resize each image to.
    Returns:
      images: a numpy array of all the images stacked together.
    """
    
    def __init__(self, base, dsize=(214, 214)):

        self.base = base
        self.dsize = dsize

    def load_images_into_array(self):
      images = []
      if os.path.isdir(self.base):
          files = os.listdir(self.base) #list the names of the images contained in the folder(base)
          for file in files:
            img = cv2.imread(os.path.join(self.base,file)) #load each image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=self.dsize, interpolation=cv2.INTER_AREA) #resize each image 
            images.append(img)
      images = np.stack(images)
      return images
