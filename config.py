from easydict import EasyDict as edict
import json
from os.path import expanduser

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 32
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
## initialize G
config.TRAIN.n_epoch_init = 10
config.TRAIN.save_init_every = 5
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

config.TRAIN.use_vgg = False
config.TRAIN.use_weighted_mse = False
config.TRAIN.coe = 0.7

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 20
config.TRAIN.save_every = 2
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
config.TRAIN.gan_loss_lambda = 1e-04

## train set location
config.TRAIN.hr_img_path = expanduser('~/disk3/240_frames_imgs_QP37/label/')
config.TRAIN.lr_img_path = expanduser('~/disk3/240_frames_imgs_QP37/input/')
config.TRAIN.hevc_split_txt_path = expanduser('~/disk3/240_frames_imgs_QP37/txt/')
#config.TRAIN.pred_img_path = expanduser('~/disk3/data/rec_yuv_QP37/pred_img/')
config.TRAIN.img_stride_W = 30
config.TRAIN.img_stride_H = 30
config.TRAIN.img_H = 64
config.TRAIN.img_W = 64
config.TRAIN.img_C = 1

config.TRAIN.summaries_dir = '/tmp/srgan_train/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
