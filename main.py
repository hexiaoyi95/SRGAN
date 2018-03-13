#! /usr/bin/python
# -*- coding: utf8 -*-


import os, time, pickle, random, time
from datetime import datetime
import numpy as np

from time import localtime, strftime
import logging, scipy
import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from datetime import datetime
cur_date =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
use_vgg = config.TRAIN.use_vgg
use_weighted_mse = config.TRAIN.use_weighted_mse
multi_loss = config.TRAIN.multi_loss
ni = int(np.ceil(np.sqrt(batch_size)))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def read_all_imgs_with_heatmap_and_crop(img_list, img_path='', txt_path='',  n_threads=32):
    """Returns cropped sub images of all images in array by given path and name of each image file"""
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_sub_imgs_with_heatmap_fn, img_path=img_path, txt_path=txt_path, sub_img_w=config.TRAIN.img_W,
                                          sub_img_h=config.TRAIN.img_H, stride_w = config.TRAIN.img_stride_W,
                                          stride_h = config.TRAIN.img_stride_H)
        imgs.extend(np.concatenate(b_imgs))
        print('read %d from %s and %s' % (len(imgs), img_path, txt_path))
    return imgs
def read_all_imgs_and_crop(img_list, path='', n_threads=32):
    """Returns cropped sub images of all images in array by given path and name of each image file"""
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_sub_imgs_fn, path=path, sub_img_w=config.TRAIN.img_W,
                                          sub_img_h=config.TRAIN.img_H, stride_w = config.TRAIN.img_stride_W,
                                          stride_h = config.TRAIN.img_stride_H)
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

        print('read %d from %s'% (len(weight_array), path))

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
    #make sure thr hr_img_path and lr_img_path hava the same imgs
    train_lr_img_list = list()
    for i in train_hr_img_list:
        name,ext = os.path.splitext(i)
        part1, part2 = name.rsplit('_',1)
        train_lr_img_list.append(part1 + '_QP%d' % qp + '_' + part2 + ext)
    if use_weighted_mse:
        hevc_split_txt_list = [ i.rsplit('.',1)[0] + '.txt' for i in train_hr_img_list ]

    train_pred_img_list = list()
    #for i in train_hr_img_list:
    #    filename = os.path.splitext(i)[0]
    #    pred_img_dir, pred_img_num = filename.rsplit('_',1)
    #    train_pred_img_list.append(os.path.join(pred_img_dir,"{}_pred.bmp".format(5)))

    #train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    #valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    #valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs_and_crop(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = read_all_imgs_and_crop(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    #train_lr_imgs = read_all_imgs_with_heatmap_and_crop(train_lr_img_list, img_path=config.TRAIN.lr_img_path, txt_path=config.TRAIN.hevc_split_txt_path, n_threads=32)

    if use_weighted_mse:
        train_weight_arrays = read_all_split_txt(hevc_split_txt_list, path=config.TRAIN.hevc_split_txt_path, n_threads=32)

    #train_pred_imgs = read_all_imgs_and_crop(train_pred_img_list, path=config.TRAIN.pred_img_path, n_threads=32)

    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.input_img_C], name='t_image_input_to_SRGAN_g_generator')
    t_target_image = tf.placeholder('float32', [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.target_img_C], name='t_target_image')

    if use_weighted_mse:
        t_mse_weight = tf.placeholder('float32', [batch_size, config.TRAIN.img_W, config.TRAIN.img_H, config.TRAIN.img_C], name='t_mse_weight')
    if multi_loss:
        net_output = SRGAN_g(t_image, is_train=True, reuse=False)
        net_g = net_output[2]
    else:
        net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    #net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    #_,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    #net_d.print_params(False)
    if use_vgg:
    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
        _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    #if multi_loss:
    #    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)[2]
    #else:
    #    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    #d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    #d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    #d_loss = d_loss1 + d_loss2

    #g_gan_loss = config.TRAIN.gan_loss_lambda * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    if use_weighted_mse:
        mse_loss = tl.cost.weighted_mean_squared_error(net_g.outputs , t_target_image, t_mse_weight, coe=config.TRAIN.coe, is_mean=True)
    elif multi_loss:
        mse_loss = 0.2 * tl.cost.mean_squared_error(net_output[0].outputs , t_target_image, is_mean=True) \
                        + 0.3 * tl.cost.mean_squared_error(net_output[1].outputs, t_target_image, is_mean=True) \
                        + 0.5 *tl.cost.mean_squared_error(net_output[2].outputs, t_target_image, is_mean=True)
    else:
        mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
        
        #orig_mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    if use_vgg:
        vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    #g_loss = mse_loss + g_gan_loss

    if use_vgg:
        g_loss += vgg_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    #d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    with tf.variable_scope('clip_beta'):
        clip_beta_v = tf.Variable(clip_beta/lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    #g_optim_init = tf.train.GradientDescentOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)
   # opt = tf.train.GradientDescentOptimizer(lr_v)
   # gradients_and_vars = opt.compute_gradients(mse_loss, g_vars)
   # gradients = [ i[0] for i in gradients_and_vars]
   # clipped_gradients, norm =  tf.clip_by_global_norm(gradients, clip_beta)
   # g_optim_init = opt.apply_gradients(zip(clipped_gradients, g_vars))
    ## SRGAN
    #g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    #d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	###============================= SUMMARY  ===============================###
    tf.summary.scalar('mse_loss', mse_loss)
    #tf.summary.scalar('g_gan_loss', g_gan_loss)
    #tf.summary.scalar('d_loss', d_loss)
    #tf.summary.scalar('g_loss', g_loss)


    merged = tf.summary.merge_all()
    train_writer_init = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/g_init', sess.graph)
    train_writer_gan = tf.summary.FileWriter(config.TRAIN.summaries_dir + '/' + cur_date + '/gan', sess.graph)
    tl.layers.initialize_global_variables(sess)

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)

    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    if use_vgg:
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted( npz.items() ):
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
    #sample_sub_hr_imgs = tl.prepro.threading_data(sample_hr_img_list, fn=get_sub_imgs_fn,
    #                                              path=config.TRAIN.hr_img_path,
    #                                              sub_img_w=config.TRAIN.img_W,
    #                                              sub_img_h=config.TRAIN.img_H)
    #print('sample HR sub-image:', sample_hr_imgs.min(), sample_hr_imgs.max())
    #sample_sub_lr_imgs = tl.prepro.threading_data(sample_lr_img_list, fn=get_sub_imgs_fn,
    #                                              path=config.TRAIN.lr_img_path,
    #                                              sub_img_w=config.TRAIN.img_W,
    #                                              sub_img_h=config.TRAIN.img_H
    #                                              )
    #print('sample LR sub-image:', sample_sub_lr_imgs.min(), sample_sub_lr_imgs.max())

    #print np.asarray(sample_hr_imgs).shape
    #tl.vis.save_images(np.asarray(sample_hr_imgs), [ni, ni], save_dir_ginit+'/_train_sample_label.png')
    #tl.vis.save_images(np.asarray(sample_lr_imgs), [ni, ni], save_dir_ginit+'/_train_sample_input.png')
    #tl.vis.save_images(np.asarray(sample_hr_imgs), [ni, ni], save_dir_gan+'/_train_sample_label.png')
    #tl.vis.save_images(np.asarray(sample_lr_imgs), [ni, ni], save_dir_gan+'/_train_sample_input.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    #sess.run(tf.assign(lr_v, lr_init))
    #print(" ** fixed learning rate: %f (for init G)" % lr_init)
    total_iter = 0
    for epoch in range(0, n_epoch_init+1):
        if epoch !=0 and (epoch % decay_every_init == 0):
            new_lr_decay = lr_decay_init ** (epoch // decay_every_init)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            #sess.run(tf.assign(clip_beta_v, clip_beta / new_lr_decay)) 
            log = " ** new learning rate: %f (for init G)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for init G)" % (lr_init, decay_every_init, lr_decay_init)
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
            #b_imgs_pred = list()
            for i in range(batch_size):
                b_imgs_hr.append(train_hr_imgs[random_idx[idx + i]])
                b_imgs_lr.append(train_lr_imgs[random_idx[idx + i]])
                #b_imgs_pred.append(train_pred_imgs[random_idx[idx + i]])
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
                errM, _ , summary= sess.run([mse_loss, g_optim_init, merged],
                                            {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            train_writer_init.add_summary(summary, total_iter) 
            total_mse_loss += errM
            n_iter += 1
            total_iter +=1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)

        ## quick evaluation on train set
       # if (epoch != 0) and (epoch % config.TRAIN.save_init_every == 0):
       #     out = sess.run(net_g_test.outputs, {t_image: sample_lr_imgs})#; print('gen sub-image:', out.shape, out.min(), out.max())
       #     print("[*] save images")
       #     tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % config.TRAIN.save_init_every == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init_epoch{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
	total_iter = 0
    train_g = True
    train_d = True
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
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
            #b_imgs_pred = list()
            if use_weighted_mse:
                b_weight_arrays = list()
            for i in range(batch_size):
                b_imgs_hr.append(train_hr_imgs[random_idx[idx + i]])
                b_imgs_lr.append(train_lr_imgs[random_idx[idx + i]])
                if use_weighted_mse:
                    b_weight_arrays.append(train_weight_arrays[random_idx[idx + i]])
            if train_d:
                errD, _  = sess.run([d_loss, d_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
            ## update G
            if use_vgg:
                errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            else:
                if train_g:
                    if use_weighted_mse:
                        errG, errOrigM, errM, errA , summary, _  = sess.run([g_loss, orig_mse_loss, mse_loss, g_gan_loss, merged, g_optim],{t_image: b_imgs_lr, t_target_image: b_imgs_hr, t_mse_weight: b_weight_arrays})
                        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f origin_mse: %.8f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errOrigM, errA))
                    else:
                        errG, errM, errA , summary, _  = sess.run([g_loss, mse_loss, g_gan_loss, merged, g_optim],{t_image: b_imgs_lr, t_target_image: b_imgs_hr})
                     
                        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
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

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
      #  if (epoch != 0) and (epoch % config.TRAIN.save_every == 0):
      #      out = sess.run(net_g_test.outputs, {t_image: sample_lr_imgs})#; print('gen sub-image:', out.shape, out.min(), out.max())
      #      print("[*] save images")
      #      tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % config.TRAIN.save_every == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}_epoch{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)

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
    imid = 64 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir+'/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir+'/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir+'/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir+'/valid_bicubic.png')

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
    else:
        raise Exception("Unknow --mode")
