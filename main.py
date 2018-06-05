#! /usr/bin/python
# -*- coding: utf8 -*-


import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np

from time import localtime, strftime
import logging, scipy
import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorlayer as tl
import matplotlib.pyplot as plt
import pickle
from tensorflow.contrib import slim

sys.path.insert(0, './tf_models/research/slim')

from nets import resnet_v2
from model import *
from utils import *
from config import config, log_config
from datetime import datetime

cur_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

###====================== HYPER-PARAMETERS ===========================###
qp = config.TRAIN.QP
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
clip_beta = config.TRAIN.clip_beta
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
lr_decay_init = config.TRAIN.lr_decay_init
decay_every_init = config.TRAIN.decay_every_init
decay_every_init_2 = config.TRAIN.decay_every_init_2
use_vgg = config.TRAIN.use_vgg
use_weighted_mse = config.TRAIN.use_weighted_mse
multi_loss = config.TRAIN.multi_loss
ni = int(np.ceil(np.sqrt(batch_size)))


def cal_PSNR(input_img, orig_img):
    """ Returns psnr of all images in array by given a batch of images and labels"""

    mse = np.mean(np.mean((input_img - orig_img) ** 2, axis=2, dtype=np.float64), axis=1, dtype=np.float64)

    flag_array = mse < 1e-10
    psnr = np.zeros_like(mse)
    mse[flag_array] = 1
    psnr = 10.0 * np.log10(1 / mse)

    return psnr


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def read_all_imgs_with_heatmap_and_crop(img_list, img_path='', txt_path='', n_threads=32):
    """Returns cropped sub images of all images in array by given path and name of each image file"""
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_sub_imgs_with_heatmap_fn, img_path=img_path,
                                          txt_path=txt_path, sub_img_w=config.TRAIN.img_W,
                                          sub_img_h=config.TRAIN.img_H, stride_w=config.TRAIN.img_stride_W,
                                          stride_h=config.TRAIN.img_stride_H)
        imgs.extend(np.concatenate(b_imgs))
        print('read %d from %s and %s' % (len(imgs), img_path, txt_path))
    return imgs


def read_all_imgs_and_crop(img_list, path='', n_threads=32):
    """Returns cropped sub images of all images in array by given path and name of each image file"""
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_sub_imgs_fn, path=path, sub_img_w=config.TRAIN.img_W,
                                          sub_img_h=config.TRAIN.img_H, stride_w=config.TRAIN.img_stride_W,
                                          stride_h=config.TRAIN.img_stride_H)
        imgs.extend(np.concatenate(b_imgs))
        print('read %d from %s' % (len(imgs), path))
    return imgs


def read_all_split_txt(txt_list, path='', n_threads=32):
    """Return weight in array by given path and name of each txt"""
    weight_array = []
    for idx in range(0, len(txt_list), n_threads):
        b_txt_list = txt_list[idx: idx + n_threads]
        b_weight_array = tl.prepro.threading_data(b_txt_list, fn=get_sub_wegiht_array_fn, path=path,
                                                  sub_img_w=config.TRAIN.img_W, sub_img_h=config.TRAIN.img_H)
        weight_array.extend(np.concatenate(b_weight_array))

        print('read %d from %s' % (len(weight_array), path))

    return weight_array


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    random.shuffle(train_hr_img_list)
    # make sure thr hr_img_path and lr_img_path hava the same imgs
    train_lr_img_list = list()
    for i in train_hr_img_list:
        name, ext = os.path.splitext(i)
        part1, part2 = name.rsplit('_', 1)
        train_lr_img_list.append(part1 + '_QP%d' % qp + '_' + part2 + ext)
    print train_lr_img_list
    if use_weighted_mse:
        hevc_split_txt_list = [i.rsplit('.', 1)[0] + '.txt' for i in train_hr_img_list]

    train_pred_img_list = list()
    # for i in train_hr_img_list:
    #    filename = os.path.splitext(i)[0]
    #    pred_img_dir, pred_img_num = filename.rsplit('_',1)
    #    train_pred_img_list.append(os.path.join(pred_img_dir,"{}_pred.bmp".format(5)))

    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs_and_crop(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = read_all_imgs_and_crop(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    # train_lr_imgs = read_all_imgs_with_heatmap_and_crop(train_lr_img_list, img_path=config.TRAIN.lr_img_path, txt_path=config.TRAIN.hevc_split_txt_path, n_threads=32)

    if use_weighted_mse:
        train_weight_arrays = read_all_split_txt(hevc_split_txt_list, path=config.TRAIN.hevc_split_txt_path,
                                                 n_threads=32)

    # train_pred_imgs = read_all_imgs_and_crop(train_pred_img_list, path=config.TRAIN.pred_img_path, n_threads=32)

    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.input_img_C],
                             name='t_image_input_to_SRGAN_g_generator')
    t_target_image = tf.placeholder('float32',
                                    [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.target_img_C],
                                    name='t_target_image')

    if use_weighted_mse:
        t_mse_weight = tf.placeholder('float32',
                                      [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.img_C],
                                      name='t_mse_weight')
    if multi_loss:
        net_output = SRGAN_g(t_image, is_train=True, reuse=False)
        net_g = net_output[2]
    else:
        net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    # net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    # _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    # net_d.print_params(False)
    if use_vgg:
        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0,
                                                    align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0,
                                                     align_corners=False)  # resize_generate_image_for_vgg

        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
        _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    # if multi_loss:
    #    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)[2]
    # else:
    #    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    # d_loss = d_loss1 + d_loss2

    # g_gan_loss = config.TRAIN.gan_loss_lambda * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    if use_weighted_mse:
        mse_loss = tl.cost.weighted_mean_squared_error(net_g.outputs, t_target_image, t_mse_weight,
                                                       coe=config.TRAIN.coe, is_mean=True)
    elif multi_loss:
        mse_loss = 0.2 * tl.cost.mean_squared_error(net_output[0].outputs, t_target_image, is_mean=True) \
                   + 0.3 * tl.cost.mean_squared_error(net_output[1].outputs, t_target_image, is_mean=True) \
                   + 0.5 * tl.cost.mean_squared_error(net_output[2].outputs, t_target_image, is_mean=True)
    else:
        mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)

        # orig_mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    if use_vgg:
        vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    # g_loss = mse_loss + g_gan_loss

    if use_vgg:
        g_loss += vgg_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    # d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    with tf.variable_scope('clip_beta'):
        clip_beta_v = tf.Variable(clip_beta / lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # g_optim_init = tf.train.GradientDescentOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)
    # opt = tf.train.GradientDescentOptimizer(lr_v)
    # gradients_and_vars = opt.compute_gradients(mse_loss, g_vars)
    # gradients = [ i[0] for i in gradients_and_vars]
    # clipped_gradients, norm =  tf.clip_by_global_norm(gradients, clip_beta)
    # g_optim_init = opt.apply_gradients(zip(clipped_gradients, g_vars))
    ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    ###============================= SUMMARY  ===============================###
    tf.summary.scalar('mse_loss', mse_loss)
    # tf.summary.scalar('g_gan_loss', g_gan_loss)
    # tf.summary.scalar('d_loss', d_loss)
    # tf.summary.scalar('g_loss', g_loss)


    merged = tf.summary.merge_all()
    train_writer_init = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/g_init', sess.graph)
    train_writer_gan = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/gan', sess.graph)
    tl.layers.initialize_global_variables(sess)

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                    network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                                     network=net_g)

    # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    if use_vgg:
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
        # net_vgg.print_params(False)
        # net_vgg.print_layers()
    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_hr_imgs = train_hr_imgs[0:batch_size]
    sample_lr_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    # sample_sub_hr_imgs = tl.prepro.threading_data(sample_hr_img_list, fn=get_sub_imgs_fn,
    #                                              path=config.TRAIN.hr_img_path,
    #                                              sub_img_w=config.TRAIN.img_W,
    #                                              sub_img_h=config.TRAIN.img_H)
    # print('sample HR sub-image:', sample_hr_imgs.min(), sample_hr_imgs.max())
    # sample_sub_lr_imgs = tl.prepro.threading_data(sample_lr_img_list, fn=get_sub_imgs_fn,
    #                                              path=config.TRAIN.lr_img_path,
    #                                              sub_img_w=config.TRAIN.img_W,
    #                                              sub_img_h=config.TRAIN.img_H
    #                                              )
    # print('sample LR sub-image:', sample_sub_lr_imgs.min(), sample_sub_lr_imgs.max())

    # print np.asarray(sample_hr_imgs).shape
    # tl.vis.save_images(np.asarray(sample_hr_imgs), [ni, ni], save_dir_ginit+'/_train_sample_label.png')
    # tl.vis.save_images(np.asarray(sample_lr_imgs), [ni, ni], save_dir_ginit+'/_train_sample_input.png')
    # tl.vis.save_images(np.asarray(sample_hr_imgs), [ni, ni], save_dir_gan+'/_train_sample_label.png')
    # tl.vis.save_images(np.asarray(sample_lr_imgs), [ni, ni], save_dir_gan+'/_train_sample_input.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    # sess.run(tf.assign(lr_v, lr_init))
    # print(" ** fixed learning rate: %f (for init G)" % lr_init)
    total_iter = 0
    for epoch in range(0, n_epoch_init + 1):
        if epoch != 0 and (epoch % decay_every_init == 0):
            new_lr_decay = lr_decay_init ** (epoch // decay_every_init)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            # sess.run(tf.assign(clip_beta_v, clip_beta / new_lr_decay))
            log = " ** new learning rate: %f (for init G)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for init G)" % (
            lr_init, decay_every_init, lr_decay_init)
            print(log)
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        fix_length = len(train_hr_imgs) - len(train_hr_imgs) % batch_size
        random_idx = list(range(0, fix_length))
        random.shuffle(random_idx)
        for idx in range(0, fix_length, batch_size):
            step_time = time.time()
            b_imgs_hr = list()
            b_imgs_lr = list()
            if use_weighted_mse:
                b_weight_arrays = list()
            # b_imgs_pred = list()
            for i in range(batch_size):
                b_imgs_hr.append(train_hr_imgs[random_idx[idx + i]])
                b_imgs_lr.append(train_lr_imgs[random_idx[idx + i]])
                # b_imgs_pred.append(train_pred_imgs[random_idx[idx + i]])
                if use_weighted_mse:
                    b_weight_arrays.append(train_weight_arrays[random_idx[idx + i]])
            ## update G
            if use_weighted_mse:
                errOrigM, errM, _, summary = sess.run([orig_mse_loss, mse_loss, g_optim_init, merged],
                                                      {t_image: b_imgs_lr, t_target_image: b_imgs_hr,
                                                       t_mse_weight: b_weight_arrays})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, weighted_mse: %.8f, origin mse: %.8f" %
                      (epoch, n_epoch_init, n_iter, time.time() - step_time, errM, errOrigM))
            else:
                errM, _, summary = sess.run([mse_loss, g_optim_init, merged],
                                            {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            train_writer_init.add_summary(summary, total_iter)
            total_mse_loss += errM
            n_iter += 1
            total_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
        epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        # if (epoch != 0) and (epoch % config.TRAIN.save_init_every == 0):
        #     out = sess.run(net_g_test.outputs, {t_image: sample_lr_imgs})#; print('gen sub-image:', out.shape, out.min(), out.max())
        #     print("[*] save images")
        #     tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % config.TRAIN.save_init_every == 0):
            tl.files.save_npz(net_g.all_params,
                              name=checkpoint_dir + '/g_{}_init_epoch{}.npz'.format(tl.global_flag['mode'], epoch),
                              sess=sess)

            ###========================= train GAN (SRGAN) =========================###
        total_iter = 0
    train_g = True
    train_d = True
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        fix_length = len(train_hr_imgs) - len(train_hr_imgs) % batch_size
        random_idx = list(range(0, fix_length))
        random.shuffle(random_idx)
        for idx in range(0, fix_length, batch_size):
            step_time = time.time()

            b_imgs_hr = list()
            b_imgs_lr = list()
            # b_imgs_pred = list()
            if use_weighted_mse:
                b_weight_arrays = list()
            for i in range(batch_size):
                b_imgs_hr.append(train_hr_imgs[random_idx[idx + i]])
                b_imgs_lr.append(train_lr_imgs[random_idx[idx + i]])
                if use_weighted_mse:
                    b_weight_arrays.append(train_weight_arrays[random_idx[idx + i]])
            if train_d:
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
            ## update G
            if use_vgg:
                errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],
                                                     {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (
                epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            else:
                if train_g:
                    if use_weighted_mse:
                        errG, errOrigM, errM, errA, summary, _ = sess.run(
                            [g_loss, orig_mse_loss, mse_loss, g_gan_loss, merged, g_optim],
                            {t_image: b_imgs_lr, t_target_image: b_imgs_hr, t_mse_weight: b_weight_arrays})
                        print(
                        "Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f origin_mse: %.8f adv: %.6f)" % (
                        epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errOrigM, errA))
                    else:
                        errG, errM, errA, summary, _ = sess.run([g_loss, mse_loss, g_gan_loss, merged, g_optim],
                                                                {t_image: b_imgs_lr, t_target_image: b_imgs_hr})

                        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)" % (
                        epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
                train_writer_gan.add_summary(summary, total_iter)
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1
            total_iter += 1
            discr_loss_ratio = errD / errG
            # if discr_loss_ratio < 1e-1 and train_d:
            #     train_g = True
            #     train_d = False
            # if discr_loss_ratio > 5e-1 and not train_d:
            #     train_d = True
            #     train_g = True
            # if discr_loss_ratio >1e1 and train_g:
            #     train_d = True
            #     train_g = False

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
        epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        #  if (epoch != 0) and (epoch % config.TRAIN.save_every == 0):
        #      out = sess.run(net_g_test.outputs, {t_image: sample_lr_imgs})#; print('gen sub-image:', out.shape, out.min(), out.max())
        #      print("[*] save images")
        #      tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % config.TRAIN.save_every == 0):
            tl.files.save_npz(net_g.all_params,
                              name=checkpoint_dir + '/g_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch),
                              sess=sess)
            tl.files.save_npz(net_d.all_params,
                              name=checkpoint_dir + '/d_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch),
                              sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (
    size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


def train_multiple():
    ## create folders to save result images and trained model
    # save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    # tl.files.exists_or_mkdir(save_dir_ginit)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    target_models_num = 4
    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.bmp', printable=False))
    random.shuffle(train_hr_img_list)
    # make sure thr hr_img_path and lr_img_path hava the same imgs
    train_lr_img_list = train_hr_img_list
    # train_lr_img_list = list()
    # for i in train_hr_img_list:
    #     name, ext = os.path.splitext(i)
    #     part1, part2 = name.rsplit('_', 1)
    #     train_lr_img_list.append(part1 + '_QP%d' % qp + '_' + part2 + ext)

    train_hr_imgs = read_all_imgs_and_crop(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # train_lr_imgs = read_all_imgs_and_crop(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    train_lr_imgs = read_all_imgs_with_heatmap_and_crop(train_lr_img_list, img_path=config.TRAIN.lr_img_path,
                                                        txt_path=config.TRAIN.hevc_split_txt_path, n_threads=32)
    # train_data_psnr = cal_PSNR(np.asarray(train_hr_imgs), np.asarray(train_lr_imgs))
    # n, bins, patches = plt.hist(train_data_psnr, 1000, density=True, facecolor='g')
    # print bins
    # plt.axis([20, 70, 0, 0.1])
    # plt.grid(True)
    # plt.ylabel('PSNR')
    # plt.savefig('his.png')
    # exit(-1)


    g_1 = tf.Graph()
    with g_1.as_default():
        sess_1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        t_image_1 = tf.placeholder('float32', [None, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.input_img_C],
                                   name='t_image_input_1')
        t_target_image = tf.placeholder('float32',
                                        [None, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.target_img_C],
                                        name='t_target_image')
        with tf.variable_scope('learning_rate_1'):
            lr_v_1 = tf.Variable(lr_init, trainable=False)

    g_2 = tf.Graph()
    with g_2.as_default():
        sess_2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        t_labels = tf.placeholder(tf.int32, [batch_size], name='classify_labels')
        t_image_2 = tf.placeholder('float32',
                                   [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, 1],
                                   name='t_image_input_2')
        with tf.variable_scope('learning_rate_2'):
            lr_v_2 = tf.Variable(lr_init, trainable=False)

    ###===============Define resnet and load ckpt==============================
    with g_2.as_default():
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, endpoints = resnet_v2.resnet_v2_50(inputs=t_image_2, num_classes=target_models_num,
                                                       reuse=False, is_training=True)
        checkpoint_exclude_scopes = ["resnet_v2_50/conv1", "resnet_v2_50/logits"]
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables('resnet_v2_50'):
           excluded = False
           for exclusion in exclusions:
               if var.op.name.startswith(exclusion):
                   excluded = True
                   break
           if not excluded:
               variables_to_restore.append(var)

        init_fn = slim.assign_from_checkpoint_fn(
           '/home/l301/DataSet2/tensorflow_model/resnet_v2_50.ckpt',
           variables_to_restore)
        
        saver = tf.train.Saver(var_list=slim.get_model_variables('resnet_v2_50'))

        softmax_loss = tl.cost.cross_entropy(logits, t_labels, 'softmax_loss')
        softmax_opt = tf.train.AdamOptimizer(lr_v_2, beta1=beta1).minimize(softmax_loss,
                                                                           var_list=slim.get_model_variables(
                                                                               'resnet_v2_50'))

        tf.summary.scalar('softmax_loss', softmax_loss)
        merged_2 = tf.summary.merge_all()
        train_writer_2 = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/train_multiple_2',
                                               sess_2.graph)
        sess_2.run(tf.global_variables_initializer())

        # Training from scratch
        # init_fn(sess_2)

        # Fine-tune
        saver.restore(sess_2, tf.train.latest_checkpoint('./checkpoint_vrcnn_fusion_resiClass'))

    with g_1.as_default():

        nets = [VRCNN_fusion(t_image_1, is_train=True, reuse=False, scope='vrcnn_fusion' + '_' + str(i)) for i in
                range(target_models_num)]
        nets_4_test = [VRCNN_fusion(t_image_1, is_train=False, reuse=True, scope='vrcnn_fusion' + '_' + str(i)) for i in
                       range(target_models_num)]

        mse_loss = [tl.cost.mean_squared_error(nets[i].outputs, t_target_image, is_mean=True) for i in
                    range(target_models_num)]
        train_opts = []
        # train_g_and_vars = []
        # train_apply_ops = []
        for i in range(len(nets)):
            vars = tl.layers.get_variables_with_name('vrcnn_fusion' + '_' + str(i), True, True)
            opt = tf.train.AdamOptimizer(lr_v_1, beta1=beta1).minimize(mse_loss[i], var_list=vars)
            train_opts.append(opt)
            # opt = tf.train.AdamOptimizer(lr_v_1, beta1=beta1)
            # g_and_vars = opt.compute_gradients(mse_loss[i], vars)
            # train_apply_ops.append(opt.apply_gradients(g_and_vars))
            # train_opts.append(opt)
            # train_g_and_vars.append(g_and_vars)
            tf.summary.scalar('mse_loss' + '_' + str(i), mse_loss[i])
        merged_1 = tf.summary.merge_all()
        train_writer_1 = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/train_multiple_1',
                                               sess_1.graph)
        sess_1.run(tf.global_variables_initializer())

        for ii, nn in enumerate(nets):
            tl.files.load_and_assign_npz(sess=sess_1,
                                         name='./checkpoint_vrcnn_fusion_resiClass' + '/{}_{}_epoch{}.npz'.format(
                                             str(ii),
                                             tl.global_flag['mode'], 40), network=nn)
            # tl.files.load_and_assign_npz(sess=sess_1,
            #                              name='/home/l301/DataSet2/tensorflow_model/checkpoint_data04_netVRCNN-fusion-QP22/g_srgan_init_epoch40.npz',
            #                              network=nn)
    ## If your machine have enough memory, please pre-load the whole train set.
    # the last 10000 sub-images used to validate

    fix_length = len(train_hr_imgs) - len(train_hr_imgs) % batch_size

    random_idx = list(range(0, fix_length))
    random.shuffle(random_idx)

    # random_idx = pickle.load(open('./train_dataset_split0'))
    labels_by_rmac = np.load(open('/home/l301/temp_labels.npy'))
    test_num = 10000
    test_rand_idx = random_idx[:test_num]
    total_train_rand_idx = random_idx[test_num:]

    # with open('train_dataset_idx_' + cur_date, 'w') as pk_fp:
    #     pickle.dump(total_train_rand_idx, pk_fp)
    #     pk_fp.close()

    train_rand_idx = total_train_rand_idx
    # train_rand_idx = total_train_rand_idx[:len(total_train_rand_idx) / 2]
    # train_rand_idx = total_train_rand_idx[len(total_train_rand_idx) / 2:]
    print "Using the last 10000 sub-images to validate, the first half of rest are used to train"
    train_labels = []
    ###========================= start training====================###
    total_iter = 0
    for epoch in range(0, n_epoch_init + 1):
        if epoch != 0 and (epoch % decay_every_init == 0):
            new_lr_decay = lr_decay_init ** (epoch // decay_every_init)
            sess_2.run(tf.assign(lr_v_2, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for init G)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess_1.run(tf.assign(lr_v_1, lr_init))
            sess_2.run(tf.assign(lr_v_2, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for init G)" % (
            lr_init, decay_every_init, lr_decay_init)
            print(log)

        second_stage_epoch = epoch - config.TRAIN.second_stage_start_at
        if second_stage_epoch != 0 and (second_stage_epoch % decay_every_init_2 == 0):
            new_lr_decay = lr_decay_init ** (second_stage_epoch // decay_every_init_2)
            sess_1.run(tf.assign(lr_v_1, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for init G)" % (lr_init * new_lr_decay)
            print(log)
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        for idx in range(0, len(train_rand_idx), batch_size):
            if idx + batch_size > len(train_rand_idx):
                continue
            b_imgs_hr = list()
            b_imgs_lr = list()
            b_imgs_lr_single_c = list()
            b_imgs_resi = list()
            for i in range(batch_size):
                b_imgs_hr.append(train_hr_imgs[train_rand_idx[idx + i]])
                b_imgs_lr.append(train_lr_imgs[train_rand_idx[idx + i]])

                if config.TRAIN.input_img_C == 1:
                    b_imgs_lr_single_c.append(train_lr_imgs[train_rand_idx[idx + i]])
                    b_imgs_resi.append(
                        abs(train_lr_imgs[train_rand_idx[idx + i]] - train_hr_imgs[train_rand_idx[idx + i]]))
                else:
                    b_imgs_lr_single_c.append(
                        train_lr_imgs[train_rand_idx[idx + i]][:, :, 0].reshape(config.TRAIN.img_H,
                                                                                config.TRAIN.img_W, 1))
                    b_imgs_resi.append(
                        abs(train_lr_imgs[train_rand_idx[idx + i]][:,:,0] \
                            - train_hr_imgs[train_rand_idx[idx + i]][:,:,0]).reshape(config.TRAIN.img_H,
                                                                                config.TRAIN.img_W, 1))

            # 00
            if epoch == 0:
                forward_results = []
                err = []
                step_time = time.time()
                for net_index in range(len(nets)):

                    err_, net_output = sess_1.run([mse_loss[net_index], nets[net_index].outputs],
                                                    {t_image_1: b_imgs_lr, t_target_image: b_imgs_hr})
                    err.append(err_)
                    forward_results.append(cal_PSNR(net_output, b_imgs_hr))

                print("Generating labels, Epoch [%2d/%2d] %4d forward time: %4.4fs, all nets mse: %s " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, str(err)))

                # print forward_results
                # shape : 2,20
                my_labels = np.asarray(forward_results).argmax(axis=0)
                my_labels = my_labels.reshape((batch_size,))
                train_labels.extend(list(my_labels))
                # print my_labels
                continue

            step_time = time.time()

            my_labels = train_labels[idx:idx + batch_size]
            # print 'labels: ', my_labels

            # 01

            # input_psnr = cal_PSNR(np.asarray(b_imgs_lr_single_c), np.asarray(b_imgs_hr))
            # my_labels = np.zeros_like(input_psnr, dtype=np.int32)
            # my_labels[input_psnr > 37] = 1
            # my_labels = my_labels.reshape((batch_size,))
            # print my_labels

            # 10
            # my_labels = np.asarray([labels_by_rmac[train_rand_idx[idx+i]] for i in range(batch_size)], dtype=np.int32)
            # my_labels = my_labels.reshape((batch_size,))
            if not epoch > config.TRAIN.first_stage_stop:
                err_softmax, _, summary, probs = sess_2.run([softmax_loss, softmax_opt, merged_2, endpoints['predictions']],
                                                        {t_image_2: b_imgs_resi, t_labels: my_labels})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, softmax: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, err_softmax))
                train_writer_2.add_summary(summary, total_iter)
            else:

                probs = sess_2.run(endpoints['predictions'],{t_image_2: b_imgs_resi})

            flags = probs.argmax(axis=1)
            # flags = probs > 1.0/3
            # print 'predictions: ', flags

            if second_stage_epoch > 0:
                for net_index in range(len(nets)):
                    step_time = time.time()
                    single_net_imgs_lr = [b_imgs_lr[i] for i, f in enumerate(flags) if f == net_index]
                    single_net_imgs_hr = [b_imgs_hr[i] for i, f in enumerate(flags) if f == net_index]
                    # single_net_imgs_lr = [b_imgs_lr[i] for i, f in enumerate(flags) if f[net_index]]
                    # single_net_imgs_hr = [b_imgs_hr[i] for i, f in enumerate(flags) if f[net_index]]

                    if len(single_net_imgs_hr) == 0:
                        continue
                    err, _, summary = sess_1.run([mse_loss[net_index], train_opts[net_index], merged_1],
                                                 {t_image_1: single_net_imgs_lr, t_target_image: single_net_imgs_hr})
                    print("Epoch [%2d/%2d] %4d time %4.4fs, %4dth net: mse: %.8f " % (
                        second_stage_epoch, n_epoch_init - config.TRAIN.second_stage_start_at, n_iter,
                        time.time() - step_time, net_index, err))
                    train_writer_1.add_summary(summary, total_iter)

            # total_mse_loss += err
            n_iter += 1
            total_iter += 1
        # log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time,
        #                                                         total_mse_loss/n_iter)
        # print(log)

        ## quick evaluation on train set
        if (second_stage_epoch > 0) and (second_stage_epoch % config.TRAIN.test_every == 0):
            test_result = [[] for i in nets_4_test]
            input_result = []
            for test_idx in range(0, test_num, batch_size):
                if test_idx + batch_size > test_num:
                    continue
                test_lr_imgs = [train_lr_imgs[i] for i in test_rand_idx[test_idx:test_idx + batch_size]]
                test_hr_imgs = [train_hr_imgs[i] for i in test_rand_idx[test_idx:test_idx + batch_size]]
                if config.TRAIN.input_img_C != 1:
                    test_input_imgs = [train_lr_imgs[i][:, :, 0].reshape(config.TRAIN.img_H, config.TRAIN.img_W, 1) \
                                       for i in test_rand_idx[test_idx:test_idx + batch_size]]
                else:
                    test_input_imgs = test_lr_imgs
                input_result.extend(cal_PSNR(np.asarray(test_input_imgs), np.asarray(test_hr_imgs)))
                for test_net_idx, test_net in enumerate(nets_4_test):
                    out = sess_1.run(test_net.outputs, {t_image_1: test_lr_imgs})
                    test_result[test_net_idx].extend(cal_PSNR(np.asarray(out), np.asarray(test_hr_imgs)))
            input_average_psnr = np.asarray(input_result, dtype=np.float32).mean()
            average_gains = [np.asarray(i, dtype=np.float32).mean() - input_average_psnr for i in test_result]
            test_summary = tf.Summary()
            for i in range(len(average_gains)):
                test_summary.value.add(tag='Average Gain ' + str(i), simple_value=average_gains[i])
            train_writer_1.add_summary(test_summary, second_stage_epoch)
            print average_gains, input_average_psnr

        ## save model
        if (epoch != 0) and (epoch % config.TRAIN.save_init_every == 0):
            saver.save(sess_2, os.path.join('./', checkpoint_dir, 'resnet_v2_50'), global_step=epoch)

        if (second_stage_epoch > 0) and (second_stage_epoch % config.TRAIN.save_init_every == 0):
            for ii, nn in enumerate(nets):
                tl.files.save_npz(nn.all_params,
                                  name=checkpoint_dir + '/{}_{}_epoch{}.npz'.format(str(ii),
                                                                                    tl.global_flag['mode'],
                                                                                    second_stage_epoch), sess=sess_1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    elif tl.global_flag['mode'] == 'train_multiple':
        train_multiple()
    else:
        raise Exception("Unknow --mode")
