import torch.nn as nn
from collections import namedtuple

###########################################################################################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(100, 64 * 32, 4, 1, 0),
            nn.BatchNorm2d(64 * 32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 32, 64 * 16, 4, 1, 0),
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(),    
            
            nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 1, 0),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.ReLU()
        )

    def forward(self, noise):
        output = self.main(noise)
        return output




D_Ouputs = namedtuple('DiscriminatorOuputs', ['aux_cls', 'out'])

class Discriminator(nn.Module):
    def __init__(self, aux_cls = True):
        super(Discriminator, self).__init__()
        
        self.aux_cls = aux_cls
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64*2, 4, 2, 1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            
            nn.Conv2d(64*2, 64*4, 4, 2, 1),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            
            nn.Conv2d(64*4, 64*8, 4, 2, 1),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.ReLU()
        )
        
        
    def forward(self, img):
        
        out = self.encoder(img)
        out = self.decoder(out)
        
        return out


