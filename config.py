import torch

# hazed data path
path_hazed = "/content/gdrive/My Drive/hazed_1/"
#'C:/dataset/hazed_300'

# clear data path
path_clear = "/content/gdrive/My Drive/clear_1/"
# 'C:/dataset/clear_300'

# Is the PC has cuda
cuda_use = torch.cuda.is_available()
# which cuda to use
cuda_num = 1

# learning rate for D, the lr in Apple blog is 0.0001
d_lr = 0.0001
# learning rate for R, the lr in Apple blog is 0.0001
r_lr = 0.0001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.01
img_width = 28
img_height = 28
img_channels = 1

# synthetic image path
#syn_path = 'dataset/SynthEyes_train_data'
# real image path
#real_path = 'dataset/MPIIGaze_data'
# training result path to save
train_res_path = 'C:/유민형/개인 연구/ocr via domain adaptation/train_res'
# final_res_path = 'final_res'

# result show in 4 sample per line
pics_line = 3

# =================== training params ======================
# pre-train R times
r_pretrain = 1000
# pre-train D times
d_pretrain = 200
# train steps
train_steps = 10000

batch_size = 64
# test_batch_size = 128
# the history buffer size
buffer_size = 1
k_d = 1  # number of discriminator updates per step
k_r = 50  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 50
# save model dictionary and training dataset output result per train times
save_per = 10


# pre-training dictionary path
#ref_pre_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/R_pre.pkl'
ref_pre_path = None
#disc_pre_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/D_pre.pkl'
disc_pre_path = None

# dictionary saving path
D_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/D_%d.pkl'
R_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/R_%d.pkl'

# load trained model
#D_load_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/D_10.pkl'
#R_load_path = 'C:/유민형/개인 연구/ocr via domain adaptation/models/R_10.pkl'
D_load_path = None
R_load_path = None




