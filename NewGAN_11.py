import torch.nn as nn
from collections import namedtuple

###########################################################################################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(62, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

D_Ouputs = namedtuple('DiscriminatorOuputs', ['aux_cls', 'out'])

class Discriminator(nn.Module):
    def __init__(self, aux_cls = True):
        super(Discriminator, self).__init__()
        
        self.aux_cls = aux_cls
        
        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = 16
        down_dim = 64 * (16) ** 2
        self.encoder = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 1, 3, 1, 1))
        
        # ----- classifier ----- #
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        
    def forward(self, img):
        out = self.down(img)
        out = self.encoder(out.view(out.size(0), -1))
        out = self.decoder(out)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        
        if self.aux_cls:
            aux_out = out
            
        out = self.model(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
        if self.aux_cls:
            return D_Ouputs(aux_out, validity)
        return validity


