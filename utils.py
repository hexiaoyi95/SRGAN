import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_sub_imgs_fn(file_name, path, sub_img_w, sub_img_h, row_index=0, col_index=1):
    x = scipy.misc.imread(path + file_name, mode='F')
    x = x / 255
    h, w = x.shape[row_index], x.shape[col_index]
    results = list()
    for l_w in range(0, w - w%sub_img_w, sub_img_w):
        for l_h in range(0, h - h%sub_img_h, sub_img_h):
            sub_img = x[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
            results.append(sub_img.reshape(sub_img_h,sub_img_w,1))
    return np.asarray(results)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def get_batch_randomly(x, random_idx, start_idx, batch_size):
    batch = list()
    print len(x)
    for i in range(batch_size):
        batch.append(x[random_idx[start_idx + i]])

    return batch
