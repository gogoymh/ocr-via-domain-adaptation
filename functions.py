from torch.utils.data import Dataset
from skimage import io, transform
import os
import torch
import math
import PIL
import numpy as np
import gzip

##################################################################################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##################################################################################################################################
class normalize(object):
    # img_type: numpy
    #img = img * 1.0 / 255
    def __init__(self, mean, sigma):
        assert mean >= 0 and mean <= 1
        assert sigma > 0
        
        self.mean = mean
        self.sigma = sigma
    def __call__(self, image):
        return (image - self.mean) / self.sigma

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
        image = 1-(image.reshape(28,28,1)/255)
        image = (image*255).astype('uint8')
        image = PIL.Image.fromarray(image.reshape(28,28), "L")
        
        if self.transform:
            image = self.transform(image)
            
        return image

class jpg2tensor(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
    
    def __len__(self):
        num = 156
        return num
    
    def __getitem__(self, idx):
        file = ("%d" % (idx+1)) + '.jpg'
        img_name = os.path.join(self.path,file)
        
        image = io.imread(img_name)
        image = PIL.Image.fromarray(image)
        image = image.convert('L')
        image = np.array(image)
        
        image = image/255
        image[np.where(image > 0) and np.where(image < 0.5)] = image[np.where(image > 0) and np.where(image < 0.5)] - 0.1
        image[np.where(image >= 0.5) and np.where(image <= 1)] = image[np.where(image >= 0.5) and np.where(image <= 1)] + 0.1
        image[np.where(image > 1)] = 1
        image[np.where(image < 0)] = 0
        image = image*255
        
        image = np.reshape(image, (28, 28, 1))
        image = image.astype('uint8')
        image = PIL.Image.fromarray(image.reshape(28,28), 'L')
        
        if self.transform:
            image = self.transform(image)
            
        return image

class CenterCrop(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size
      
  def __call__(self, image):
    image = np.array(image)
    h = image.shape[0:2][0]
    w = image.shape[0:2][1]
    new_h, new_w = self.output_size
    
    top = math.floor(h/2)  - math.floor(new_h/2)
    left = math.floor(w/2) - math.floor(new_w/2)
    
    assert top > 0 and left > 0
    
    bottom = top + new_h
    right = left + new_w
    
    crop = image[top:bottom,left:right]
    crop = PIL.Image.fromarray(crop)
    
    return crop

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

class ToTensor(object):
  def __call__(self, image):
    image = image.transpose((2,0,1))
    return torch.from_numpy(image).double()