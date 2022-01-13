import argparse
import os
import numpy as np

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
from architecture_2 import Generator, Discriminator
import functions as fn

######################################################################################################################
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
#parser.add_argument("--exp", type=int, default=6, help="number of experiment")
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
generator = Generator(4, opt.channels, nb_features=30).cuda()
discriminator = Discriminator().cuda()

# Initialize weights ################################################################################################
generator.apply(fn.weights_init_normal)
discriminator.apply(fn.weights_init_normal)

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
'''
EMNIST_data = fn.EMNIST_loader("emnist-letters-train-images-idx3-ubyte.gz",
                               transform=transforms.Compose([
                                       fn.Rescale(opt.img_size),
                                       fn.ToTensor()]))
EMNIST = DataLoader(EMNIST_data, batch_size=opt.batch_size, shuffle=True)
'''
tensor_hand = fn.jpg2tensor("C:/유민형/개인 연구/ocr via domain adaptation/data/real",
                            transform=transforms.Compose([
                                    fn.CenterCrop(20),
                                    fn.Rescale(opt.img_size),                                    
                                    fn.ToTensor()]))

Handwritten = DataLoader(tensor_hand, batch_size=opt.batch_size, shuffle=True)

# Optimizers #######################################################################################################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

# Criteria #########################################################################################################
self_regularization_loss = nn.L1Loss(size_average=False)

# BEGAN hyper parameters ###########################################################################################
gamma_aux = 1
lambda_k_aux = 0.001
k_aux = 0.0
'''
gamma_ref = 0.75
lambda_k_ref = 0.001
k_ref = 0.0
'''
# Train ###########################################################################################################
for epoch in range(opt.n_epochs):
    for i in range(1950):

        # Configure input
        syn_imgs = Variable(Handwritten.__iter__().next()).float().cuda() #Variable(imgs.type(Tensor))
        #real_imgs = Variable(Handwritten.__iter__().next()).float().cuda()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (32, opt.latent_dim))))

        # Generate a batch of images
        aux_img, _ = generator(z)

        # ---------- Generator ---------- #
        optimizer_G.zero_grad()
        
        # Auxiliary
        g_loss_aux = torch.mean(torch.abs(discriminator(aux_img) - aux_img))
        '''
        # Refine
        g_loss_adv = torch.mean(torch.abs(discriminator(ref_img) - ref_img))
        g_loss_reg = self_regularization_loss(ref_img, aux_img)
        g_loss_reg_scale = torch.mul(g_loss_reg, opt.delta)
        g_loss_ref = g_loss_adv + g_loss_reg_scale
        '''
        # Sum Up
        g_loss = g_loss_aux# + g_loss_ref
        
        # Backprop and update
        g_loss.backward()
        optimizer_G.step()
        
        # ----------- Discriminator ----------- #
        optimizer_D.zero_grad()

        # Auxiliary
        d_real_aux = discriminator(syn_imgs)
        d_fake_aux = discriminator(aux_img.detach())

        d_loss_real_aux = torch.mean(torch.abs(d_real_aux - syn_imgs))
        d_loss_fake_aux = torch.mean(torch.abs(d_fake_aux - aux_img.detach()))
        d_loss_aux = d_loss_real_aux - k_aux * d_loss_fake_aux
        '''
        # Refine
        d_real_ref = discriminator(real_imgs)
        d_fake_ref = discriminator(ref_img.detach())

        d_loss_real_ref = torch.mean(torch.abs(d_real_ref - real_imgs))
        d_loss_fake_ref = torch.mean(torch.abs(d_fake_ref - ref_img.detach()))
        d_loss_ref = d_loss_real_ref - k_ref * d_loss_fake_ref
        '''
        # Sum up
        d_loss = d_loss_aux #+ d_loss_ref
        
        # Backprop and update
        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights
        # ----------------

        diff_aux = torch.mean(gamma_aux * d_loss_real_aux - d_loss_fake_aux)
        #diff_ref = torch.mean(gamma_ref * d_loss_real_ref - d_loss_fake_ref)

        # Update weight term for fake samples
        k_aux = k_aux + lambda_k_aux * diff_aux.item()
        #k_ref = k_ref + lambda_k_ref * diff_ref.item()
        k_aux = min(max(k_aux, 0), 1)  # Constraint to interval [0, 1]
        #k_ref = min(max(k_ref, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M_aux = (d_loss_real_aux + torch.abs(diff_aux)).item()
        #M_ref = (d_loss_real_ref + torch.abs(diff_ref)).item()

        # --------------
        # Log Progress
        # --------------
        print("-" * 120)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D_aux loss: %f] [G_aux loss: %f] -- M_aux: %f, k_aux: %f"
            % (epoch, opt.n_epochs, i, 1950, d_loss_aux.item(), g_loss_aux.item(), M_aux, k_aux)
        )
        '''
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D_ref loss: %f] [G_ref loss: %f] -- M_ref: %f, k_ref: %f"
            % (epoch, opt.n_epochs, i, len(EMNIST), d_loss_ref.item(), g_loss_ref.item(), M_ref, k_ref)
        )
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D_tot loss: %f] [G_tot loss: %f] -- M_tot: %f, k_tot: %f"
            % (epoch, opt.n_epochs, i, len(EMNIST), d_loss.item(), g_loss.item(), (M_aux+M_ref), (k_aux+k_ref))
        )
        '''

        batches_done = epoch * 1950 + i
        if batches_done % opt.sample_interval == 0:
            print("\n")
            print("Saved!")
            print("\n")
            save_image(aux_img.data[:25], "images/test_hand2/%d_aux.png" % batches_done, nrow=5, normalize=True)
            #save_image(ref_img.data[:25], "images/test/%d_ref.png" % batches_done, nrow=5, normalize=True)
            
            save_image(d_real_aux.data[:25], "images/test_hand2/%d_aux_d.png" % batches_done, nrow=5, normalize=True)
            #save_image(d_real_ref.data[:25], "images/test/%d_ref_d.png" % batches_done, nrow=5, normalize=True)
            
            save_image(syn_imgs.data[:25], "images/test_hand2/%d_aux_gt.png" % batches_done, nrow=5, normalize=True)











