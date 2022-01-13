import torch.nn as nn
import argparse
from collections import namedtuple

###########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()

###########################################################################################################################
class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=100):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features)
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        convs = self.convs(x)
        sum = convs + x
        output = self.relu(sum)
        return output

GeneratorOuputs = namedtuple('GeneratorOuputs', ['aux_img', 'output'])

class Generator(nn.Module):
    def __init__(self, block_num, in_features, nb_features=100, aux_dis=True):
        super(Generator, self).__init__()
        
        self.aux_dis = aux_dis
        # ---------- Generator ---------- #
        
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        # ---------- refiner ---------- #
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(nb_features)
        )

        blocks = []
        for i in range(block_num):
            blocks.append(ResnetBlock(nb_features, nb_features))

        self.resnet_blocks = nn.Sequential(*blocks)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(nb_features, in_features, 1, 1, 0),
            nn.Tanh()
        )
        

    def forward(self, noise):
        # ---------- Generator ---------- #
        out = self.l1(noise)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        
        if self.aux_dis:
            aux_img = img
        
        # ----------- Refiner ---------- #
        conv_1 = self.conv_1(img)
        res_block = self.resnet_blocks(conv_1)
        output = self.conv_2(res_block)
        
        if self.aux_dis:
            return GeneratorOuputs(aux_img, output)
        
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 128, 4, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 128 * (opt.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 128, self.down_size, self.down_size))
        return out

