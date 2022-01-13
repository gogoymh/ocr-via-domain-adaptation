import torch.nn as nn

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

class Refiner(nn.Module):
    def __init__(self, block_num, in_features, nb_features=100, aux_dis=True):
        super(Refiner, self).__init__()
        
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
        

    def forward(self, img):
        # ----------- Refiner ---------- #
        conv_1 = self.conv_1(img)
        res_block = self.resnet_blocks(conv_1)
        output = self.conv_2(res_block)
        
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = 16
        down_dim = 64 * (16) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 1, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out

