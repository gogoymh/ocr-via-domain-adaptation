import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn
from torch.utils.data import DataLoader #, TensorDataset
#from torchvision import datasets
from torch.autograd import Variable

#import torch.nn as nn
#import torch.nn.functional as F
import torch

######################################################################################################################
from architecture_new_4 import Refiner, Discriminator1, Discriminator2
import functions as fn

######################################################################################################################
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=6, help="number of experiment")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of sampling images")
parser.add_argument("--delta", type=float, default=0.01, help="Scale factor of refine loss")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# Initialize generator and discriminator ############################################################################
generator = Refiner(4, opt.channels, nb_features=30).cuda()
discriminator1 = Discriminator1().cuda()
discriminator2 = Discriminator2().cuda()


# Initialize weights ################################################################################################
generator.apply(fn.weights_init_normal)
discriminator1.apply(fn.weights_init_normal)

# Configure data loader #############################################################################################
os.makedirs("C:/유민형/개인 연구/ocr via domain adaptation/data", exist_ok=True)
'''
EMNIST = DataLoader(
    datasets.EMNIST(
        "C:/유민형/개인 연구/ocr via domain adaptation/data",
        split="letters",
        train=True,
        download=False,
        transform=transforms.Compose([
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
'''
EMNIST_data = fn.EMNIST_loader("emnist-letters-train-images-idx3-ubyte.gz",
                               transform=transforms.Compose([
                                       fn.Rescale(opt.img_size),
                                       fn.ToTensor()]))
EMNIST = DataLoader(EMNIST_data, batch_size=opt.batch_size, shuffle=True)

tensor_hand = fn.jpg2tensor("C:/유민형/개인 연구/ocr via domain adaptation/data/real",
                            transform=transforms.Compose([
                                    fn.CenterCrop(20),
                                    fn.Rescale(opt.img_size),                                    
                                    fn.ToTensor()]))

Handwritten = DataLoader(tensor_hand, batch_size=opt.batch_size, shuffle=True)

# Optimizers #######################################################################################################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

# Criteria #########################################################################################################
self_regularization_loss = nn.L1Loss(size_average=False)
local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)

# BEGAN hyper parameters ###########################################################################################
gamma = 0.75
lambda_k = 0.001
k = 0.0

# Train ###########################################################################################################
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(EMNIST):

        # Configure input
        syn_img = Variable(imgs.type(Tensor))
        real_img = Variable(Handwritten.__iter__().next()).float().cuda()

        # Generate a batch of images
        ref_img = generator(syn_img)

        # ---------- Refiner ---------- #
        optimizer_G.zero_grad()
        
        # BEGAN
        g_loss_BEGAN = torch.sum(torch.abs(discriminator1(ref_img) - ref_img))
        
        # simGAN
        d_ref_pred = discriminator2(ref_img).view(-1, 2)
        d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()
        g_loss_adv = local_adversarial_loss(d_ref_pred, d_real_y)
        
        g_loss_reg = self_regularization_loss(ref_img, syn_img)
        g_loss_reg_scale = torch.mul(g_loss_reg, opt.delta)
        
        g_loss_simGAN = g_loss_adv + g_loss_reg_scale
        
        # Sum Up
        g_loss = g_loss_BEGAN + g_loss_simGAN
        
        # Backprop and update
        g_loss.backward()
        optimizer_G.step()
        
        # ----------- Discriminator ----------- #
        optimizer_D1.zero_grad()

        # BEGAN
        d_real = discriminator1(real_img)
        d_fake = discriminator1(ref_img.detach())

        d_loss_real = torch.sum(torch.abs(d_real - real_img))
        d_loss_fake = torch.sum(torch.abs(d_fake - ref_img.detach()))
        
        d_loss_BEGAN = d_loss_real - k * d_loss_fake
        
        # simGAN
        d_real_pred = discriminator2(real_img).view(-1, 2)
        d_real_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
        d_loss_real2 = local_adversarial_loss(d_real_pred, d_real_y)
        
        d_ref_pred = discriminator2(ref_img).view(-1, 2)
        d_ref_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()
        d_loss_ref = local_adversarial_loss(d_ref_pred, d_ref_y)
        
        d_loss_simGAN = d_loss_real2 + d_loss_ref
        
        # Backprop and update
        d_loss_BEGAN.backward()
        optimizer_D1.step()
        
        d_loss_simGAN.backward()
        optimizer_D2.step()

        # ----------- Parameter ----------- #
        diff = torch.sum(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).item()

        # --------------
        # Log Progress
        # --------------
        print("-" * 120)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss BEGAN: %f] [D loss simGAN: %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, opt.n_epochs, i, len(EMNIST), d_loss_BEGAN.item(), d_loss_simGAN.item(), g_loss.item(), M, k)
        )

        batches_done = epoch * len(EMNIST) + i
        if batches_done % opt.sample_interval == 0:
            save_image(ref_img.data[:25], "images/exp%d/%d_refined.png" % (opt.exp, batches_done), nrow=5, normalize=True)
            
            save_image(d_real.data[:25], "images/exp%d/%d_autoencoder.png" % (opt.exp, batches_done), nrow=5, normalize=True)











