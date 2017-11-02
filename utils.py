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

def get_sub_wegiht_array_fn(file_name, path, sub_img_w, sub_img_h, row_index=0, col_index=1):
    fp = open(path + file_name)
    lines = fp.readlines()
    last_lx, last_ly, last_rx, last_ry = lines[-1].split(' ')
    w = last_rx + 1
    h = last_ry + 1
    weight_array = np.zeros((w, h),dtype=np.float32)
    weight_array[0,0:w] = 1.0
    weight_array[0:h,0] = 1.0

    for line in lines:
        lx,ly,rx,ry,part_size,pre_mode= line.split(' ')
        weight_array[rx, ly:ry+1] = 1.0
        weight_array[lx:rx+1, ry] = 1.0

    results = list()
    for l_w in range(0, w - w % sub_img_w, sub_img_w):
        for l_h in range(0, h - h % sub_img_h, sub_img_h):
            sub_array = weight_array[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
            results.append(sub_array.reshape(sub_img_h, sub_img_w, 1))
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
