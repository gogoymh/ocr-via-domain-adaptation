import numpy as np
import gzip
#import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torch import nn
from torch.autograd import Variable
from image_history_buffer import ImageHistoryBuffer
from network import Discriminator, Refiner
from image_utils import generate_img_batch, calc_acc
import config as cfg
import os

from torch.utils.data import TensorDataset

##############################################################################################################################
def generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):
    print('=' * 50)
    print('Generating a batch of training images...')
    self.R.eval()

    pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
    generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
    print('=' * 50)

##############################################################################################################################
# ===== Label  Sign ===== #
alphabet = ["a","b","c","d","e","f","g","h","i","j","k",
            "l","m","n","o","p","q","r","s","t","u","v",
            "w","x","y","z"]

##############################################################################################################################
f_tr_i = gzip.open('emnist-letters-train-images-idx3-ubyte.gz','rb')
data = np.frombuffer(f_tr_i.read(), dtype=np.uint8, offset=16).astype(np.float32)
data = data.reshape(124800,28,28)
for i in range(124800):
    data[i] = data[i].transpose()
data = data.reshape(124800,1,28,28)/255
data = torch.from_numpy(data)
tensor_hand = TensorDataset(data)




'''
idx = 40
image = np.asarray(data[idx]).squeeze()
plt.imshow(image)
plt.show()
'''
f_tr_l = gzip.open("emnist-letters-train-labels-idx1-ubyte.gz", "rb")
labels = np.frombuffer(f_tr_l.read(), np.uint8, offset=8)
#print(alphabet[labels[idx]-1])
#f_te_i = gzip.open('emnist-letters-test-images-idx3-ubyte.gz','rb')
#data2 = np.frombuffer(f_te_i.read(), dtype=np.uint8, offset=16).astype(np.float32)
#data2 = data2.reshape(20800,image_size, image_size)
#for i in range(20800):
#    data2[i] = data2[i].transpose()
#image = np.asarray(data2[idx]).squeeze()
#plt.imshow(image)
#plt.show()
#f_te_l = gzip.open("emnist-letters-test-labels-idx1-ubyte.gz", "rb")
#labels2 = np.frombuffer(f_te_l.read(), np.uint8, offset=8)
#print(alphabet[labels2[idx]-1])
train_img = data.reshape(124800,1,28,28)/255
#test_img = data2.reshape(20800,1,28,28)/255
train_label = labels
#test_label = labels2
train_img = torch.from_numpy(train_img)
train_label = torch.from_numpy(train_label)
#test_img = torch.from_numpy(test_img)

##############################################################################################################################
# ===== Prapare handwritten data ===== #

##############################################################################################################################
# ===== build_network ===== #
print('=' * 50)
print('Building network...')
R = Refiner(4, cfg.img_channels, nb_features=64).cuda()
D = Discriminator(input_features=cfg.img_channels).cuda()

opt_R = torch.optim.Adam(R.parameters(), lr=cfg.r_lr)
opt_D = torch.optim.SGD(D.parameters(), lr=cfg.d_lr)
self_regularization_loss = nn.L1Loss(size_average=False)
local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
delta = cfg.delta

# ===== load_data ===== #
print('=' * 50)
print('Loading data...')

train = Data.TensorDataset(train_img, train_label)
syn_train_loader = Data.DataLoader(train, batch_size = cfg.batch_size, shuffle = True, pin_memory=True)
print('syn_train_batch %d' % len(syn_train_loader))

real = Data.TensorDataset()
real_loader = Data.DataLoader(real, batch_size = cfg.batch_size, shuffle = True, pin_memory=True)
print('real_batch %d' % len(real_loader))

# ===== pre_train_r ===== #
print('=' * 50)
if cfg.ref_pre_path:
    print('Loading R_pre from %s' % cfg.ref_pre_path)
    R.load_state_dict(torch.load(cfg.ref_pre_path))


# we first train the Rθ network with just self-regularization loss for 1,000 steps
print('pre-training the refiner network %d times...' % cfg.r_pretrain)

for index in range(cfg.r_pretrain):
    syn_image_batch, _ = syn_train_loader.__iter__().next()
    syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

    R.train()
    ref_image_batch = R(syn_image_batch)

    r_loss = self_regularization_loss(ref_image_batch, syn_image_batch)
    # r_loss = torch.div(r_loss, cfg.batch_size)
    r_loss = torch.mul(r_loss, delta)

    opt_R.zero_grad()
    r_loss.backward()
    opt_R.step()

    # log every `log_interval` steps
    if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
        # figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
        print('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.r_pretrain, r_loss.data[0]))

        syn_image_batch, _ = syn_train_loader.__iter__().next()
        syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)

        real_image_batch, _ = real_loader.__iter__().next()
        real_image_batch = Variable(real_image_batch, volatile=True)

        R.eval()
        ref_image_batch = R(syn_image_batch)

        figure_path = os.path.join(cfg.train_res_path, 'refined_image_batch_pre_train_%d.png' % index)
        generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                   real_image_batch.data, figure_path)
        R.train()

        print('Save R_pre to models/R_pre.pkl')
        torch.save({'epoch': index,
                    'model_state_dict': R.state_dict(),
                    'optimizer_state_dict': opt_R.state_dict(),
                    'loss': r_loss}, 'C:/유민형/개인 연구/ocr via domain adaptation/models/R_pre.pkl')

# ===== pre_train_d ===== #
print('=' * 50)
if cfg.disc_pre_path:
    print('Loading D_pre from %s' % cfg.disc_pre_path)
            
    checkpoint_pre_d = torch.load(cfg.disc_pre_path)
            
    D.load_state_dict(checkpoint_pre_d['model_state_dict'])
    opt_D.load_state_dict(checkpoint_pre_d['optimizer_state_dict'])
    d_epoch = checkpoint_pre_d['epoch']
    d_loss = checkpoint_pre_d['loss']
    print('Discriminator pre-trained %d times, D_Loss is %.4f' % ((d_epoch+1), d_loss))

# and Dφ for 200 steps (one mini-batch for refined images, another for real)
print('pre-training the discriminator network %d times...' % cfg.r_pretrain)    
    
D.train()
R.eval()
for index in range(cfg.d_pretrain):
    real_image_batch, _ = real_loader.__iter__().next()
    real_image_batch = Variable(real_image_batch).cuda(cfg.cuda_num)

    syn_image_batch, _ = syn_train_loader.__iter__().next()
    syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

    assert real_image_batch.size(0) == syn_image_batch.size(0)

    # ============ real image D ====================================================
    # D.train()
    d_real_pred = D(real_image_batch).view(-1, 2)                                    

    d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
    d_ref_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()

    acc_real = calc_acc(d_real_pred, 'real')
    d_loss_real = local_adversarial_loss(d_real_pred, d_real_y)
    # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

    # ============ syn image D ====================================================
    # R.eval()
    ref_image_batch = R(syn_image_batch)

    # D.train()
    d_ref_pred = D(ref_image_batch).view(-1, 2)

    acc_ref = calc_acc(d_ref_pred, 'refine')
    d_loss_ref = local_adversarial_loss(d_ref_pred, d_ref_y)
    # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

    d_loss = d_loss_real + d_loss_ref
    opt_D.zero_grad()
    d_loss.backward()
    opt_D.step()

    if (index % cfg.d_pre_per == 0) or (index == cfg.d_pretrain - 1):
        print('[%d/%d] (D)d_loss:%f  acc_real:%.2f%% acc_ref:%.2f%%'
              % (index, cfg.d_pretrain, d_loss.data[0], acc_real, acc_ref))

print('Save D_pre to models/D_pre.pkl')
torch.save({'epoch': index,
            'model_state_dict': D.state_dict(),
            'optimizer_state_dict': opt_D.state_dict(),
            'loss': d_loss}, 'C:/유민형/개인 연구/ocr via domain adaptation/models/D_pre.pkl')    

##############################################################################################################################
# ===== train ===== #
print('=' * 50)
print('Training...')
startpoint = 0
if cfg.R_load_path:
    checkpoint_R = torch.load(cfg.R_load_path)
            
    R.load_state_dict(checkpoint_R['model_state_dict'])
    opt_R.load_state_dict(checkpoint_R['optimizer_state_dict'])
    r_epoch = checkpoint_R['epoch']
    r_loss = checkpoint_R['loss']
    print('Refiner trained %d times, R_Loss is %.4f' % ((r_epoch+1), r_loss))
            
if cfg.D_load_path:
    checkpoint_D = torch.load(cfg.D_load_path)
            
    D.load_state_dict(checkpoint_D['model_state_dict'])
    opt_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
    d_epoch = checkpoint_D['epoch']
    d_loss = checkpoint_D['loss']
    print('Discriminator trained %d times, D_Loss is %.4f' % ((d_epoch+1), d_loss))
            
if cfg.R_load_path and cfg.D_load_path:
    checkpoint_R = torch.load(cfg.R_load_path)
    checkpoint_D = torch.load(cfg.D_load_path)
            
    r_epoch = checkpoint_R['epoch']
    d_epoch = checkpoint_D['epoch']
            
    assert r_epoch == d_epoch
    startpoint = r_epoch+1

image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                          cfg.buffer_size * 10, cfg.batch_size)

for step in range(cfg.train_steps):
    print('Step[%d/%d]' % (step, cfg.train_steps))

    # ========= train the R =========
    D.eval()
    R.train()

    for p in D.parameters():
        p.requires_grad = False

    total_r_loss = 0.0
    total_r_loss_reg_scale = 0.0
    total_r_loss_adv = 0.0
    total_acc_adv = 0.0
    
    for index in range(cfg.k_r):
        syn_image_batch, _ = syn_train_loader.__iter__().next()
        syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

        ref_image_batch = R(syn_image_batch)
        d_ref_pred = D(ref_image_batch).view(-1, 2)

        d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)

        acc_adv = calc_acc(d_ref_pred, 'real')

        r_loss_reg = self_regularization_loss(ref_image_batch, syn_image_batch)
        r_loss_reg_scale = torch.mul(r_loss_reg, delta)
        # r_loss_reg_scale = torch.div(r_loss_reg_scale, cfg.batch_size)

        r_loss_adv = local_adversarial_loss(d_ref_pred, d_real_y)
        # r_loss_adv = torch.div(r_loss_adv, cfg.batch_size)

        r_loss = r_loss_reg_scale + r_loss_adv

        opt_R.zero_grad()
        opt_D.zero_grad()
        r_loss.backward()
        opt_R.step()

        total_r_loss += r_loss
        total_r_loss_reg_scale += r_loss_reg_scale
        total_r_loss_adv += r_loss_adv
        total_acc_adv += acc_adv

    mean_r_loss = total_r_loss / cfg.k_r
    mean_r_loss_reg_scale = total_r_loss_reg_scale / cfg.k_r
    mean_r_loss_adv = total_r_loss_adv / cfg.k_r
    mean_acc_adv = total_acc_adv / cfg.k_r

    print('(R)r_loss:%.4f r_loss_reg:%.4f, r_loss_adv:%f(%.2f%%)'
          % (mean_r_loss.data[0], mean_r_loss_reg_scale.data[0], mean_r_loss_adv.data[0], mean_acc_adv))

# ========= train the D =========
R.eval()
D.train()
for p in D.parameters():
    p.requires_grad = True

for index in range(cfg.k_d):
    real_image_batch, _ = real_loader.__iter__().next()
    syn_image_batch, _ = syn_train_loader.__iter__().next()
    assert real_image_batch.size(0) == syn_image_batch.size(0)

    real_image_batch = Variable(real_image_batch).cuda(cfg.cuda_num)
    syn_image_batch = Variable(syn_image_batch).cuda(cfg.cuda_num)

    ref_image_batch = R(syn_image_batch)

    # use a history of refined images
    half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
    image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy())

    if len(half_batch_from_image_history):
        torch_type = torch.from_numpy(half_batch_from_image_history)
        v_type = Variable(torch_type).cuda(cfg.cuda_num)
        ref_image_batch[:cfg.batch_size // 2] = v_type

        d_real_pred = D(real_image_batch).view(-1, 2)
        d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
        d_loss_real = local_adversarial_loss(d_real_pred, d_real_y)
        # d_loss_real = torch.div(d_loss_real, cfg.batch_size)
        acc_real = calc_acc(d_real_pred, 'real')

        d_ref_pred = D(ref_image_batch).view(-1, 2)
        d_ref_y = Variable(torch.ones(d_ref_pred.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
        d_loss_ref = local_adversarial_loss(d_ref_pred, d_ref_y)
        # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)
        acc_ref = calc_acc(d_ref_pred, 'refine')

        d_loss = d_loss_real + d_loss_ref

        D.zero_grad()
        d_loss.backward()
        opt_D.step()

        print('(D)d_loss:%.4f real_loss:%.4f(%.2f%%) refine_loss:%.4f(%.2f%%)'
              % (d_loss.data[0] / 2, d_loss_real.data[0], acc_real, d_loss_ref.data[0], acc_ref))

if step % cfg.save_per == 0:
    print('Save two model dict.')
    torch.save({'epoch': step,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': opt_D.state_dict(),
                'loss': d_loss}, cfg.D_path % step)
    torch.save({'epoch': step,
                'model_state_dict': R.state_dict(),
                'optimizer_state_dict': opt_R.state_dict(),
                'loss': r_loss}, cfg.R_path % step)

    real_image_batch, _ = real_loader.__iter__().next()
    syn_image_batch, _ = syn_train_loader.__iter__().next()
    real_image_batch = Variable(real_image_batch, volatile=True).cuda(cfg.cuda_num)
    syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)

    R.eval()
    ref_image_batch = R(syn_image_batch)
    generate_batch_train_image(syn_image_batch, ref_image_batch, real_image_batch, step_index=step)








##############################################################################################################################

























