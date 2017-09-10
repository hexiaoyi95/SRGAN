from easydict import EasyDict as edict
import json
from os.path import expanduser

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 2
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

config.TRAIN.use_vgg = False

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 30
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = expanduser('~/disk3/data/imgs_QP37/label/')
config.TRAIN.lr_img_path = expanduser('~/disk3/data/imgs_QP37/input/')
config.TRAIN.img_H = 128
config.TRAIN.img_W = 128
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
