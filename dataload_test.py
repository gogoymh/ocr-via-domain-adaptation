import math
from skimage import transform, io
import numpy as np
import gzip
import torch
from torch.utils.data import TensorDataset, Dataset
import matplotlib.pyplot as plt
import os
import PIL
'''
class EMNIST_loader(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        
        f_tr_i = gzip.open(path,'rb')
        data = np.frombuffer(f_tr_i.read(), dtype=np.uint8, offset=16).astype(np.float32)
        data = data.reshape(124800,28,28)
        for i in range(124800):
            data[i] = data[i].transpose()
        self.data = data
    
    def __len__(self):
        num = 124800
        return num
    
    def __getitem__(self, idx):
        
        image = self.data[idx]
        image = image.reshape(124800,1,28,28)/255
        
        if self.transform:
            image = self.transform(image)
            
        return image



class Rescale(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size
      
  def __call__(self, image):
    h, w = image.shape[0:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = math.floor(self.output_size * h / w), self.output_size
      else:
        new_h, new_w = self.output_size, math.floor(self.output_size * w / h)
    else:
      new_h, new_w = self.output_size
      
    new_h, new_w = int(new_h), int(new_w)
    
    image = transform.resize(image, (new_h, new_w))
    
    return image
'''
f_tr_i = gzip.open('emnist-letters-train-images-idx3-ubyte.gz','rb')
data = np.frombuffer(f_tr_i.read(), dtype=np.uint8, offset=16).astype(np.float32)
data = data.reshape(124800,28,28)
for i in range(124800):
    data[i] = data[i].transpose()

image = data[0]
plt.imshow(image)

data = data.reshape(124800,1,28,28)/255

data = torch.from_numpy(data)
tensor_hand = TensorDataset(data)

