import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.autograd import Variable

import numpy as np
import torch

######################################################################################################################
from NewGAN_17 import Generator, Discriminator
import functions as fn
from image_history_buffer import ImageHistoryBuffer

######################################################################################################################
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=22, help="number of experiment")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of sampling images")
parser.add_argument("--delta", type=float, default=0.0001, help="Scale factor of refine loss")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# Initialize generator and discriminator ############################################################################
generator = Generator(4, 1, 32).cuda()
discriminator = Discriminator().cuda()

# Configure data loader #############################################################################################
EMNIST_data = fn.EMNIST_loader("emnist-letters-train-images-idx3-ubyte.gz",
                               transform=transforms.Compose([
                                       transforms.Resize(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))])) #
EMNIST = DataLoader(EMNIST_data, batch_size=opt.batch_size, shuffle=True)

tensor_hand = fn.jpg2tensor("C:/유민형/개인 연구/ocr via domain adaptation/data/real2",
                            transform=transforms.Compose([
                                    transforms.CenterCrop(15),
                                    transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])) #

Handwritten = DataLoader(tensor_hand, batch_size=opt.batch_size, shuffle=True)

# Optimizers #######################################################################################################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

# Criteria #########################################################################################################
self_regularization_loss = nn.L1Loss(size_average=False)
local_adversarial_loss = nn.CrossEntropyLoss()

# BEGAN hyper parameters ###########################################################################################
gamma = 0.75
lambda_k = 0.001
k = 0.0

# Train ###########################################################################################################
image_history_buffer = ImageHistoryBuffer((0, 1, 32, 32), 12800 * 10, 64)
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(EMNIST):

        syn_imgs = Variable(imgs.type(Tensor))
        real_imgs = Variable(Handwritten.__iter__().next()).float().cuda()
        
        for refine in range(1):
            # ---------- Refiner ---------- #
            optimizer_G.zero_grad()
            
            ref_imgs = generator(syn_imgs)
            
            D_ref_imgs, D_ref_y = discriminator(ref_imgs)
            
            # BEGAN
            g_loss_began = self_regularization_loss(D_ref_imgs, ref_imgs)
            
            # Similarity            
            g_loss_reg = self_regularization_loss(ref_imgs, syn_imgs)
            g_loss_reg_scale = torch.mul(g_loss_reg, opt.delta)
            
            g_loss = g_loss_began + g_loss_reg_scale
            
            g_loss.backward(retain_graph=True)
            optimizer_G.step()

        # --------- Discriminator --------- #

        optimizer_D.zero_grad()
        
        # buffer
        half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
        image_history_buffer.add_to_image_history_buffer(ref_imgs.cpu().data.numpy())

        if len(half_batch_from_image_history):
            torch_type = torch.from_numpy(half_batch_from_image_history)
            v_type = Variable(torch_type).cuda()
            ref_imgs[:opt.batch_size // 2] = v_type
        
        # BEGAN
        D_real_imgs, D_real_y = discriminator(real_imgs)
        D_ref_imgs, D_ref_y = discriminator(ref_imgs)

        d_loss_real = self_regularization_loss(D_real_imgs, real_imgs)
        d_loss_fake = self_regularization_loss(D_ref_imgs, ref_imgs)
        d_loss_began = d_loss_real - k * d_loss_fake
        
        d_loss = d_loss_began 

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

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
            "[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, opt.n_epochs, i, len(EMNIST), d_loss.item(), g_loss.item(), M, k)
        )

        batches_done = epoch * len(EMNIST) + i
        if batches_done % opt.sample_interval == 0:
            save_image(syn_imgs.data[:25], "images/exp%d/%d_0sample.png" % (opt.exp, batches_done), nrow=5, normalize=True)
            
            save_image(ref_imgs.data[:25], "images/exp%d/%d_1refined.png" % (opt.exp, batches_done), nrow=5, normalize=True)
            #save_image(D_ref_imgs.data[:25], "images/exp%d/%d_2autoencoder.png" % (opt.exp, batches_done), nrow=5, normalize=True)
            
            save_image(real_imgs.data[:25], "images/exp%d/%d_3real.png" % (opt.exp, batches_done), nrow=5, normalize=True)            
            #save_image(D_real_imgs.data[:25], "images/exp%d/%d_4autoencoder.png" % (opt.exp, batches_done), nrow=5, normalize=True)











