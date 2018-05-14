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

def get_sub_imgs_with_heatmap_fn(file_name, img_path,txt_path, sub_img_w, sub_img_h, stride_w, stride_h, row_index=0, col_index=1):
    x = scipy.misc.imread(img_path + file_name, mode='F')
    if os.path.basename(file_name).split('.')[1] == 'png':
        x = x/ 1023.
    else:
        x = x/ 255.
    h, w = x.shape[row_index], x.shape[col_index]
    #make heatmap
    fp = open(txt_path + file_name.rsplit('.',1)[0] + '.txt')
    lines = fp.readlines()
    weight_array = np.zeros((h, w),dtype=np.float32)
    #case 1
    #weight_array[0:2,0:w] = 1.0
    #weight_array[0:h,0:2] = 1.0
    
    for line in lines:
        int_list = [ int(i) for i in line.split(' ')]
        ly,lx,ry,rx,part_size,pre_mode= int_list
        #case 1
        #weight_array[rx-1:rx+2, ly:ry+1] = 1.0
        #weight_array[lx:rx+1, ry-1:ry+2] = 1.0

        #case 2
        #max_depth = 4
        #value = 256/np.power(2,4) * (sub_img_h/(ry-ly)) - 1
        #weight_array[lx:rx+1, ly:ry+1] = value / 255.

        #case 3
        value = np.mean(x[lx:rx+1, ly:ly+1])
        weight_array[lx:rx+1, ly:ry+1] = value 

    results = list()
    for l_w in range(0, w - w%sub_img_w, stride_w):
        for l_h in range(0, h - h%sub_img_h, stride_h):
            if l_h + sub_img_h > h:
                l_h = h - sub_img_h
            if l_w + sub_img_w > w:
                l_w = w - sub_img_w
            sub_img = x[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
            sub_heatmap = weight_array[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
            results.append(np.stack((sub_img,sub_heatmap),-1))
    return np.asarray(results)
    #return weight_array

def get_sub_imgs_fn(file_name, path, sub_img_w, sub_img_h, stride_w, stride_h, row_index=0, col_index=1):
    x = scipy.misc.imread(path + file_name, mode='F')
    if os.path.basename(file_name).split('.')[1] == 'png':
        x = x/ 1023.
    else:
        x = x/ 255.
    #x = x / (255. / 2.)
    #x = x - 1.0
    h, w = x.shape[row_index], x.shape[col_index]
    results = list()
    for l_w in range(0, w - w%sub_img_w, stride_w):
        for l_h in range(0, h - h%sub_img_h, stride_h):
            if l_h + sub_img_h > h:
                l_h = h - sub_img_h
            if l_w + sub_img_w > w:
                l_w = w - sub_img_w
            sub_img = x[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
            results.append(sub_img.reshape(sub_img_h,sub_img_w,1))
    return np.asarray(results)

def get_sub_wegiht_array_fn(file_name, path, sub_img_w, sub_img_h, row_index=0, col_index=1):
    fp = open(path + file_name)
    lines = fp.readlines()
    int_list = [ int(i) for i in lines[-1].split(' ')]
    last_ly, last_lx, last_ry, last_rx, part_size, pre_mode = int_list
    w = last_ry + 1
    h = last_rx + 1
    weight_array = np.zeros((h, w),dtype=np.float32)
    weight_array[0,0:w] = 1.0
    weight_array[0:h,0] = 1.0

    for line in lines:
        int_list = [ int(i) for i in line.split(' ')]
        ly,lx,ry,rx,part_size,pre_mode= int_list
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
