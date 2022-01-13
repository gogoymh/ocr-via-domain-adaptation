import torch.nn as nn
from collections import namedtuple

###########################################################################################################################
class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=32):
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

class Generator(nn.Module):
    def __init__(self, block_num, in_features, nb_features=32):
        super(Generator, self).__init__()

        self.upchannel = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU()
        )
        
        blocks = []
        for i in range(block_num):
            blocks.append(ResnetBlock(nb_features, nb_features))
        
        self.resnet = nn.Sequential(*blocks)
        
        self.downchannel = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.upchannel(x)
        output = self.resnet(output)
        output = self.downchannel(output)
        return output




D_Outputs = namedtuple('DiscriminatorOuputs', ['aux_img', 'out_cls'])

class Discriminator(nn.Module):
    def __init__(self, aux_cls = True):
        super(Discriminator, self).__init__()
        
        self.aux_cls = aux_cls
        
        self.downsample = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1), # 32*16*16
                nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
                nn.Linear(32*16*16, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                
                nn.Linear(32, 32*16*16),
                nn.BatchNorm1d(32*16*16),
                nn.LeakyReLU()
        )
        # Upsampling
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32*32*32
                nn.LeakyReLU(),
                nn.Conv2d(32, 1, 3, 1, 1) # 1*32*32
        )
        
        self.fcn = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1), # 32*16*16
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                
                nn.Conv2d(32, 16, 4, 2, 1), # 16*8*8
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                
                nn.Conv2d(16, 2, 1, 1, 0), # 2*8*8
                nn.Sigmoid()
        )
        
        
    def forward(self, x):
        out = self.downsample(x)
        out = self.fc(out.view(out.size(0),-1))
        out = self.upsample(out.view(64, 32, 16, 16))
        
        if self.aux_cls :
            aux_img = out
        
        out = self.fcn(out)
        out_cls = out.view(out.size(0), -1, 2)
        
        if self.aux_cls :
            return D_Outputs(aux_img, out_cls)
        
        return out


