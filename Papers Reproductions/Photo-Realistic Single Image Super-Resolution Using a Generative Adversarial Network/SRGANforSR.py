import numpy as np
import os
import time
import scipy
from scipy.misc import imresize
import argparse

import pickle, random
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import time


from easydict import EasyDict as ed
import json
from tensorlayer.prepro import *


def crop_sub_img_fn(img, is_random=True):

    x = crop(img, wrg=384, hrg=384, is_random=is_random)
    x = x / 127.5 - 1
    return x


def downsample_fn(img):

    x = imresize(img, [96, 96], interp='bicubic', mode=None)
    x = x / 127.5 - 1
    return x


""" configure part """

config = ed()
config.TRAIN = ed()

# Adam
config.TRAIN.batch_size = 4
config.TRAIN.learning_init = 1e-4
config.TRAIN.beta = 0.9

# initialize generator
config.TRAIN.n_epoch_init = 100

# adversarial_leaning(SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.learning_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

# train set location
config.TRAIN.hr_img_path = r'C:\Users\Desktop\srdata\DIV2K_train_HR'
config.TRAIN.lr_img_path = r'C:\Users\Desktop\srdata\DIV2K_train_LR_bicubic\X4'

# test set location
config.VALID = ed()

config.VALID.hr_img_path = r'C:\Users\Desktop\srdata\DIV2K_valid_HR'
config.VALID.lr_img_path = r'C:\Users\Desktop\srdata\DIV2K_valid_LR_bicubic/X4'


def log_config(filename, configure):
    with open(filename, 'w') as f:
        f.write("========================\n")
        f.write(json.dumps(configure, indent=4))
        f.write("========================\n")


""" Network model part """


def SRGAN_generator(input_image, is_train=True, reuse=False):
    """
    Generator in Photo-Realistic Single Image Super-Resolution using a Generative Adversarial Network
    feature maps(n) and stride(s) feature maps(n) and stride(s)
    """
    w_init = tf.random_normal_initializer(stddev=0.2)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)

    with tf.variable_scope('SRGAN_generator', reuse=reuse):
        # tl.layers.set_name_reuse(reuse)  # remove for TL 1.8.0
       
        net = InputLayer(input_image, name='in')
        tensor = tl.layers.Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                                  W_init=w_init, name='n64s1/c')(net)
        temp = tensor

        # 8 residual blocks
        for i in range(16):
            tensor_n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, padding='SAME', W_init=w_init,
                              b_init=b_init, name='n64s1/c1/%s' % i)(tensor)
            tensor_n = BatchNormLayer(tensor_n, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=gamma_init, name='n64s1/b1/%s' % i)
            tensor_n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, padding='SAME', W_init=w_init,
                              b_init=b_init, name='n64s1/c2/%s' % i)(tensor_n)
            tensor_n = BatchNormLayer(tensor_n, is_train=is_train, gamma_init=gamma_init, name='n64s1/b2/%s' % i)
            tensor_n = ElementwiseLayer([tensor, tensor_n], tf.add, name='b_residual_add/%s' % i)
            tensor = tensor_n

        tensor = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, padding='SAME', W_init=w_init,
                        b_init=b_init, name='n64s1/c/m')(tensor)
        tensor = BatchNormLayer(tensor, is_train=is_train, gamma_init=gamma_init, name='n64s1/b/m')
        tensor = ElementwiseLayer([tensor, temp], tf.add, name='add3')
        # 8 residual blocks end

        # deconvolution
        
        tensor = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                        padding='SAME', W_init=w_init, name='n256s1/1')(tensor)
        tensor = SubpixelConv2d(2, None, act=tf.nn.relu, name='pixelshufflerx2/1')(tensor)

        tensor = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                        padding='SAME', W_init=w_init, name='n64s1/2')(tensor)
        tensor = SubpixelConv2d(2, None, act=tf.nn.relu, name='pixelshufflex2/2')(tensor)

        tensor = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                        padding='SAME', W_init=w_init, name='out')(tensor)

        return tensor


def SRGAN_discriminator(input_image, is_training=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.2)
    b_init = None
    gamma_init = tf.random_normal_initializer(1.0, stddev=0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    filter_num = 64
    with tf.variable_scope('SRGAN_discriminator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_image, name='input/image')
        net_h0 = Conv2d(n_filter=filter_num, filter_size=(4, 4), strides=(2, 2), act=lrelu,
                        padding='SAME', W_init=w_init, name='h0/c')(net_in)

        net_h1 = Conv2d(n_filter=filter_num*2, filter_size=(4, 4), strides=(2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h1/c')(net_h0)
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h1/bn')

        net_h2 = Conv2d(n_filter=filter_num*4, filter_size=(4, 4), strides=(2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h2/c')(net_h1)
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h2/bn')

        net_h3 = Conv2d(n_filter=filter_num*8, filter_size=(4, 4), strides=(2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h3/c')(net_h2)
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h3/bn')

        net_h4 = Conv2d(n_filter=filter_num*16, filter_size=(4, 4), strides=(2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h4/c')(net_h3)
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h4/bn')

        net_h5 = Conv2d(n_filter=filter_num*32, filter_size=(4, 4), strides=(2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h5/c')(net_h4)
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h5/bn')

        net_h6 = Conv2d(n_filter=filter_num*16, filter_size=(1, 1), strides=(1, 1), act=None,
                        padding='SAME', W_init=w_init, name='h6/c')(net_h5)
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h6/bn')

        net_h7 = Conv2d(n_filter=filter_num*8, filter_size=(1, 1), strides=(1, 1), act=None,
                        padding='SAME', W_init=w_init, name='h7/c')(net_h6)
        net_h7 = BatchNormLayer(net_h7, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(n_filter=filter_num*2, filter_size=(1, 1), strides=(1, 1), act=None,
                     padding='SAME', W_init=w_init, name='reg/c')(net_h7)
        net = BatchNormLayer(net, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='reg/bn')
        net = Conv2d(n_filter=filter_num*2, filter_size=(3, 3), strides=(1, 1), act=None,
                     padding='SAME', W_init=w_init, name='reg/c2')(net)
        net = BatchNormLayer(net, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='reg/bn2')
        net = Conv2d(n_filter=filter_num*8, filter_size=(3, 3), strides=(1, 1), act=None,
                     padding='SAME', W_init=w_init, name='reg/c3')(net)
        net = BatchNormLayer(net, act=lrelu, is_train=is_training, gamma_init=gamma_init, name='reg/bn3')

        net_h8 = ElementwiseLayer([net_h7, net], tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

        return net_ho, logits


def Vgg_19_simple_api(input_image, reuse):
    VGG_MEAN = [103.939, 116.779, 123.68]
    # turn input RGB images into BGR format
    with tf.variable_scope('VGG19', reuse=reuse):
        start_time = time.time()
        print('build the model')
        # convert RGB to BGR
        input_image = input_image * 255

        red, green, blue = tf.split(input_image, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        vgg_bgr = tf.concat([blue-VGG_MEAN[0], green-VGG_MEAN[1], red-VGG_MEAN[2]], axis=3)
        assert input_image.get_shape().as_list()[1:] == [224, 224, 3]

        net_in = InputLayer(vgg_bgr, name='input')
        # build convolutional network

        """conv1"""
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv1_1')(net_in)
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv1_2')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')(network)
        '''conv2'''
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv2_1')(network)
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv2_2')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')(network)
        '''conv3'''
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv3_1')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv3_2')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv3_3')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv3_4')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        '''conv4'''
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv4_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv4_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv4_3')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv4_4')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')(network)
        '''conv5'''
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv5_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv5_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv5_3')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', name='conv5_4')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')(network)
        convol = network
        """fc6-8"""
        network = FlattenLayer(name='flatten')(network)
        network = DenseLayer(n_units=4096, act=tf.nn.relu, name='fc6')(network)
        network = DenseLayer(n_units=4096, act=tf.nn.relu, name='fc7')(network)
        network = DenseLayer(n_units=1000, act=tf.identity, name='fc8')(network)
        print('finish the build %fs' % (time.time() - start_time))
        return network, convol


""" Hyper-parameters """

# get parameters
batch_size = config.TRAIN.batch_size
learning_init = config.TRAIN.learning_init
beta = config.TRAIN.beta

# initialize G
n_epoch_init = config.TRAIN.n_epoch_init

# adversarial learning
n_epoch = config.TRAIN.n_epoch
learning_decay = config.TRAIN.learning_decay
decay_every = config.TRAIN.decay_every

normalize_batch_size = int(np.sqrt(batch_size))

""" main part """


def train():
    # create folders to save results images and trained model
    save_dir_graph_init = 'sample/{}_graph_init'.format(tl.global_flag['mode'])
    save_dir_gan = 'sample/{}_gan'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_graph_init)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = 'checkpoint'  # checkpoint_resize_convolution
    tl.files.exists_or_mkdir(checkpoint_dir)

    """ pre load data """

    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path,
                                                       regx='.*.png', printable=False))[:400]
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path,
                                                       regx='.*.png', printable=False))[:400]
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path,
                                                       regx='.*.png', printable=False))[:400]
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path,
                                                       regx='.*.png', printable=False))[:400]

    # if computer has enough memory, could pre-load whole train set
    train_hr_img = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for image in train_hr_img:
    #   print(image.shape)
    # valid_lr_img = tl.vis.read_images(valid_hr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for image in valid_lr_img:
    #   print(image.shape)
    # valid_hr_img = tl.vis.read_images(valid_lr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for image in valid_hr_img:
    #   print(image.shape)
    # train_lr_img = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    # for image in valid_hr_img:
    #   print(image.shape)
    # exit()

    """ define model """

    # train inference
    train_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='train_image_input_to_SRGAN_generator')
    train_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='train_target_image')

    # Construct the generated model and get the output of the generated model net_generator
    net_generator = SRGAN_generator(train_image, is_train=True, reuse=False)
    # Construct a discriminant network, discriminate between net_generator.output, train_target_image,
    # net_discriminator for the entire network

    net_discriminator, logist_real = SRGAN_discriminator(train_target_image, is_training=True, reuse=False)
    _, logist_fake = SRGAN_discriminator(net_generator.outputs, is_training=True, reuse=True)

    # build VGG network
    net_generator.layer_args.print_params(False)
    net_generator.layer_args.print_layers()
    net_discriminator.layers_args.print_params(False)
    net_discriminator.layers_args.print_layers()

    # input data needs to transfer dimension.  vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    target_image_224 = tf.image.resize_images(train_target_image, [224, 224], method=0, align_corners=False)
    train_pred_image_224 = tf.image.resize_images(net_generator.outputs, [224, 224], method=0, align_corners=False)

    net_vgg, vgg_target_emb = Vgg_19_simple_api((target_image_224 + 1) / 2, reuse=False)
    _, vgg_pred_emb = Vgg_19_simple_api((net_generator + 1) / 2, reuse=True)

    # testing of training part
    net_generator_test = SRGAN_generator(train_image, False, True)

    # ========== DEFINE_TRAIN_OP =================

    discriminator_loss_1 = tl.cost.sigmoid_cross_entropy(logist_real, tf.ones_like(logist_real), name='d1')
    discriminator_loss_2 = tl.cost.sigmoid_cross_entropy(logist_fake, tf.zeros_like(logist_fake), name='d2')
    discriminator_loss = discriminator_loss_1 + discriminator_loss_2

    generator_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logist_fake, tf.ones_like(logist_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_generator.outputs, train_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_target_emb.outputs, vgg_pred_emb.outputs, is_mean=True)
    generator_loss = generator_gan_loss + mse_loss + vgg_loss

    generator_var = tl.layers.get_variables_with_name('SRGAN_generator', True, True)
    discriminator_var = tl.layers.get_variables_with_name('SRGAN_discriminator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(learning_init, trainable=False)

    generator_optimize_init = tf.train.AdamOptimizer(lr_v, beta1=beta).minimize(mse_loss, var_list=generator_var)
    generator_optimize = tf.train.AdamOptimizer(lr_v, beta1=beta).minimize(generator_loss, var_list=generator_var)
    discriminator_optimize = tf.train.AdamOptimizer(lr_v, beta1=beta).minimize(discriminator_loss,
                                                                               var_list=discriminator_var)

    # ====================== RESTORE MODEL SESS ==================

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(session)
    if tl.files.load_and_assign_npz(checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode'],
                                    network=net_generator))(session) is False:
        tl.files.load_and_assign_npz(checkpoint_dir + '/g_init_{}.npz'.format(tl.global_flag['mode'],
                                     network=net_generator))(session)
    tl.files.load_and_assign_npz(checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode'],
                                 network=net_discriminator))(session)

    # ================== load vgg params ===================

    vgg_npy_path = 'vgg19.npy'
    if not os.path.isfile(vgg_npy_path):
        print('Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg')
        exit()

    npz = np.load(vgg_npy_path, encoding='latin1').item()

    params = []
    for var in sorted(list(npz.items())):
        w = np.asarray(var[1][0])
        b = np.asarray(var[1][1])
        print(" Loading %s: %s, %s" % (var[0], w.shape, b.shape))
        params.extend([w, b])

    tl.files.assign_params(session, params, net_vgg)

    print('ok')

    # ======================== TRAIN =======================

    # use first "batch_size" of train set to have a quick test during training
    sample_imgs = train_hr_img[0:batch_size]
    # Perform random cropping to ensure that its dimensions are 384
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_img_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    # Pixel reduction
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())

    # save images
    tl.vis.save_images(sample_imgs_96, [normalize_batch_size, normalize_batch_size],
                       save_dir_graph_init + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [normalize_batch_size, normalize_batch_size],
                       save_dir_graph_init + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [normalize_batch_size, normalize_batch_size],
                       save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [normalize_batch_size, normalize_batch_size],
                       save_dir_gan + '/_train_sample_384.png')

    # ======================== initial train G =====================

    session.run(tf.assign(lr_v, learning_init))
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        n_iter = 0
        init_loss_total = 0

        for idx in range(0, len(train_hr_img), batch_size):

            b_img_384 = tl.prepro.threading_data(train_hr_img[idx:idx+batch_size], fn=crop_sub_img_fn, is_random=False)
            b_img_96 = tl.prepro.threading_data(b_img_384, fn=downsample_fn)

            _, MSE_LOSS = session.run([generator_optimize_init, mse_loss], feed_dict={train_image: b_img_96,
                                                                                      train_target_image: b_img_384})
            n_iter += 1
            init_loss_total += MSE_LOSS

        if (epoch != 0) and (epoch % 10 == 0):
            out = session.run(net_generator_test.outputs, feed_dict={train_image: sample_imgs_96})
            print('[*] save image')
            tl.vis.save_images(out, [normalize_batch_size, normalize_batch_size], save_dir_graph_init +
                               '/train_%d.png' % epoch)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_generator.all_params,
                              name=checkpoint_dir + '/g_init_{}.npz'.format(tl.global_flag['mode']))

    # ======================== train GAN ==================

    for epoch in range(0, n_epoch+1):

        if epoch != 0 and epoch % decay_every == 0:
            new_learning_decay = learning_decay ** (epoch // decay_every)
            session.run(tf.assign(lr_v, learning_decay * learning_init))
            log = '** new learning rate: %f(for GAN)' % (learning_init * new_learning_decay)
            print(log)

        elif epoch == 0:
            session.run(tf.assign(lr_v, learning_init))
            log = '** init lr: %f decay_every_init: %d, lr_decay: %f(for GAN)' % \
                  (learning_init, decay_every, learning_decay)
            print(log)

        epoch_time = time.time()
        total_discriminator_loss, total_generator_loss, n_iter = 0, 0, 0

        for idx in range(0, len(train_hr_img), batch_size):

            b_img_384 = tl.prepro.threading_data(train_hr_img[idx:idx+batch_size], fn=crop_sub_img_fn, is_random=False)
            b_img_96 = tl.prepro.threading_data(b_img_384, fn=downsample_fn)

            _, errD = session.run([discriminator_optimize, discriminator_loss],
                                  feed_dict={train_image: b_img_96, train_target_image: b_img_384})
            _, errG, errM, errV, errA = session.run([generator_optimize, generator_loss,
                                                     mse_loss, vgg_loss, generator_gan_loss],
                                                    feed_dict={train_image: b_img_96, train_target_image: b_img_384})

            total_discriminator_loss += errD
            total_generator_loss += errG
            n_iter += 1

        # evaluation on train set
        if epoch != 0 and epoch % 10 == 0:
            out = session.run(net_generator_test.outputs, feed_dict={train_image: sample_imgs_96})
            print('[*] save image')
            tl.vis.save_images(out, [normalize_batch_size, normalize_batch_size], save_dir_gan + '/train_%d' % epoch)

        if epoch != 0 and epoch % 10 == 0:
            tl.files.save_npz(net_generator.all_params,
                              name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']))(session)
            tl.files.save_npz(net_discriminator.all_params,
                              name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']))(session)


def evaluate():
    # create folders to save result images
    save_dir = 'sample/{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoints_dir = 'checkpoints'

    evaluate_hr_img_list = sorted(tl.files.load_file_list(config.VALID.hr_img_path, regx='.*.png', printable=False))
    evaluate_lr_img_list = sorted(tl.files.load_file_list(config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = tl.vis.read_images(evaluate_lr_img_list, path=config.VALID.lr_img_path, n_threads=8)
    valid_hr_imgs = tl.vis.read_images(evaluate_hr_img_list, path=config.VALID.hr_img_path, n_threads=8)

    # ==================== DEFINE MODEL =================

    image_id = 64  # the n-th image in the folder, can input any number
    valid_lr_img = valid_lr_imgs[image_id]
    valid_hr_img = valid_hr_imgs[image_id]

    valid_lr_img = (valid_lr_img / 127.5) - 1
    size = valid_lr_img.shape

    train_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_generator = SRGAN_generator(train_image, is_train=False, reuse=False)

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(session)
    tl.files.load_and_assign_npz(checkpoints_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                 network=net_generator)(session)

    output = session.run([net_generator.outputs], feed_dict={train_image: [valid_lr_img]})

    print("[*] save images")
    tl.vis.save_images(output[0], [normalize_batch_size, normalize_batch_size], save_dir + '/valid_gen.png')
    tl.vis.save_images(valid_lr_img, [normalize_batch_size, normalize_batch_size], save_dir + '/valid_lr.png')
    tl.vis.save_images(valid_hr_img, [normalize_batch_size, normalize_batch_size], save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    tl.vis.save_images(out_bicu, [normalize_batch_size, normalize_batch_size], save_dir + '/valid_out_bicu.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='srgan', help='srgan evaluate')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'srgan':
        train()

    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()

    else:
        raise Exception("Unknown --mode")
