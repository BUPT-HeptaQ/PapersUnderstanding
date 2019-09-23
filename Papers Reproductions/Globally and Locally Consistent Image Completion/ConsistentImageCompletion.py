# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import tqdm

from layer import *

# parameters
IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100


""" load part """


def load(dir_='../data/npy'):
    image_train = np.load(os.path.join(dir_, 'image_train.npy'))
    image_test = np.load(os.path.join(dir_, 'image_test.npy'))

    return image_train, image_test


""" to_npy part """

ratio = 0.95
image_size = 128

image_lib = []
# paths = glob,glob('./images/*')
image_path = glob.glob('D:/data/img_align_celedba/*jpg')

for path in image_path[:500]:
    image = cv2.imread(path)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lib.append(image)

image_lib = np.array(image_lib, dtype=np.uint8)
np.random.shuffle(image_lib)

length = int(ratio * len(image_lib))
image_train = image_lib[:length]
image_test = image_lib[length:]

if not os.path.exists('./npy'):
    os.mkdir('./npy')

np.save('./npy/image_train.npy', image_train)
np.save('./npy/image_test.npy', image_test)


""" layers part """


def convolution_layer(original_input, filter_shape, stride):
    filters = tf.get_variable(name='weight', shape=filter_shape, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    return tf.nn.conv2d(original_input, filters, [1, stride, stride, 1], padding='SAME')


def dilated_convolution_layer(original_input, filter_shape, dilation):
    filters = tf.get_variable(name='weight', shape=filter_shape, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    return tf.nn.atrous_conv2d(original_input, filters, dilation, padding='SAME')


def deconvolution_layer(original_input, filter_shape, output_shape, stride):
    filters = tf.get_variable(name='weight', shape=filter_shape, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    return tf.nn.conv2d_transpose(original_input, filters, output_shape, [1, stride, stride, 1])


def batch_normalize(original_input, is_training, decay=0.99, epsilon=0.001):
    # define the normalized operations in the training process
    def batch_normalize_train():
        # get the mean and standard deviation of the first three channels
        batch_mean, batch_var = tf.nn.moments(original_input, axes=[0, 1, 2])
        # The mean gets the mean of the momentum average, which is pop_mean * decay + batch_mean * (1-decay)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        # The standard deviation is the standard deviation of the mean of momentum:
        # pop_var * decay + batch_var * (1 - decay)
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        # tf-control_dependencies indicates that train_mean and train_var operations
        # will be carried out to update parameters before the next operation
        with tf.control_dependencies([train_mean, train_var]):
            # perform the normalization operation
            return tf.nn.batch_normalization(original_input, batch_mean, batch_var, beta, scale, epsilon)

    # define the normalization operations for the test phase
    def batch_normalize_inference():
        # the mean and standard deviation of the average momentum in
        # the training process are used as the input of the mean and standard deviation
        return tf.nn.batch_normalization(original_input, pop_mean, pop_var, beta, scale, epsilon)

    # get the last channel number, which is the number of features
    dimension = original_input.get_shape().as_list()[-1]

    # the beta, or deviation, is constructed to add to the normalized data
    # and the update operation will be carried out during the training
    beta = tf.get_variable(name='beta', shape=[dimension], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.0), trainable=True)
    # scale is constructed to multiply the normalized data and update it in the training process
    scale = tf.get_variable(name='scale', shape=[dimension], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
    # used to construct the average of momentum averages
    pop_mean = tf.get_variable(name='pop_mean', shape=[dimension], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=False)
    # used to construct the standard deviation of the momentum mean
    pop_var = tf.get_variable(name='pop_var', shape=[dimension], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.0), trainable=False)

    # If is_training equals True, carry out bn_train operation, otherwise carry out bn_inference operation
    return tf.cond(is_training, batch_normalize_train, batch_normalize_inference())


# make dimension changes for full joins
def flatten_convolution(original_input):
    num = original_input.shape[0]
    return tf.reshape(original_input, [num, -1])


# construct a fully connected function
def fully_connection_layer(original_input, output_dim):
    # gets the last dimension of the input, used to construct w
    input_dim = original_input.get_shape().as_list()[-1]
    # the dimensions of w are input_dim and output_dim
    w = tf.get_variable(name='w', shape=[input_dim, output_dim], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
    # the dimension of b is output_dim
    b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.0), trainable=True)

    return tf.add(tf.matmul(original_input, w), b)


""" Network part """


class Network:
    def __init__(self, original_input, mask, local_image, global_completion, local_completion, is_training, batch_size):
        self.batch_size = batch_size
        self.imitation = self.generator(original_input * (1 - mask), is_training)  # set the padding
        self.completion = self.imitation * mask + original_input * (1 - mask)
        # here partial completion, need combine with global

        self.real = self.discriminator(original_input, local_image, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, reuse=False)

        self.generator_loss = self.calculate_generator_loss(original_input, self.completion)
        self.discriminator_loss = self.calculate_discriminator_loss(self.real, self.fake)
        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # according to Table 2. Architecture of the image completion network. in the paper
    def generator(self, original_input, is_training):
        with tf.variable_scope('generator'):
            # 6 convolutional layers
            with tf.variable_scope('convolution1'):
                original_input = convolution_layer(original_input, [5, 5, 3, 64], 1)
                original_input = batch_normalize(original_input, is_training)  # better gradient iteration
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution2'):
                original_input = convolution_layer(original_input, [3, 3, 64, 128], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution3'):
                original_input = convolution_layer(original_input, [3, 3, 128, 128], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution4'):
                original_input = convolution_layer(original_input, [3, 3, 128, 256], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution5'):
                original_input = convolution_layer(original_input, [3, 3, 256, 256], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution6'):
                original_input = convolution_layer(original_input, [3, 3, 256, 256], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            # dilated convolutional layers
            with tf.variable_scope('dilated1'):
                original_input = dilated_convolution_layer(original_input, [3, 3, 256, 256], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('dilated2'):
                original_input = dilated_convolution_layer(original_input, [3, 3, 256, 256], 4)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('dilated3'):
                original_input = dilated_convolution_layer(original_input, [3, 3, 256, 256], 8)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('dilated4'):
                original_input = dilated_convolution_layer(original_input, [3, 3, 256, 256], 16)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            # two convolutional layers
            with tf.variable_scope('convolution7'):
                original_input = convolution_layer(original_input, [3, 3, 256, 256], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution8'):
                original_input = convolution_layer(original_input, [3, 3, 256, 256], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            # deconvolution layer
            with tf.variable_scope('deconvolution1'):
                original_input = deconvolution_layer(original_input, [4, 4, 128, 256],
                                                     [self.batch_size, 64, 64, 128], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution9'):
                original_input = convolution_layer(original_input, [3, 3, 128, 128], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            # deconvolution layer
            with tf.variable_scope('deconvolution2'):
                original_input = deconvolution_layer(original_input, [4, 4, 64, 128],
                                                     [self.batch_size, 128, 128, 64], 2)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution10'):
                original_input = convolution_layer(original_input, [3, 3, 64, 32], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.relu(original_input)

            with tf.variable_scope('convolution11'):
                original_input = convolution_layer(original_input, [3, 3, 32, 3], 1)
                original_input = batch_normalize(original_input, is_training)
                original_input = tf.nn.tanh(original_input)

        return original_input

    def discriminator(self, global_image, local_image, reuse):
        def global_discriminator(original_input):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('convolution1'):
                    original_input = convolution_layer(original_input, [5, 5, 3, 64], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution2'):
                    original_input = convolution_layer(original_input, [5, 5, 64, 128], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution3'):
                    original_input = convolution_layer(original_input, [5, 5, 128, 256], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution4'):
                    original_input = convolution_layer(original_input, [5, 5, 256, 512], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution5'):
                    original_input = convolution_layer(original_input, [5, 5, 512, 512], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution6'):
                    original_input = convolution_layer(original_input, [5, 5, 512, 512], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('fc'):
                    original_input = flatten_convolution(original_input)
                    original_input = fully_connection_layer(original_input, 1024)

            return original_input

        def local_discriminator(original_input):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('convolution1'):
                    original_input = convolution_layer(original_input, [5, 5, 3, 64], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution2'):
                    original_input = convolution_layer(original_input, [5, 5, 64, 128], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution3'):
                    original_input = convolution_layer(original_input, [5, 5, 128, 256], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution4'):
                    original_input = convolution_layer(original_input, [5, 5, 256, 512], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('convolution5'):
                    original_input = convolution_layer(original_input, [5, 5, 512, 512], 2)
                    original_input = batch_normalize(original_input, is_training)
                    original_input = tf.nn.relu(original_input)

                with tf.variable_scope('fc'):
                    original_input = flatten_convolution(original_input)
                    original_input = fully_connection_layer(original_input, 1024)

            return original_input

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_image)
            local_output = local_discriminator(local_image)

            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = fully_connection_layer(output, 1)
        return output

    def calculate_generator_loss(self, original_input, completion):
        loss = tf.nn.l2_loss(original_input - completion)  # difference of pixels

        return tf.reduce_mean(loss)

    def calculate_discriminator_loss(self, real, fake):
        alpha = 4e-4
        discriminate_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                (logits=real, labels=tf.ones_like(real)))
        discriminate_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                (logits=fake, labels=tf.zeros_like(fake)))

        return tf.add(discriminate_loss_real, discriminate_loss_fake) * alpha


""" training part """

def train_images():
    # design the placeholder
    original_input = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_input = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])

    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(original_input, mask, local_input, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    session = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    generator_train_optimizer = optimizer.minimize(model.generator_loss,
                                                   global_step=global_step, var_list=model.generator_variables)
    discriminator_train_optimizer = optimizer.minimize(model.discriminator_loss,
                                                       global_step=global_step, var_list=model.discriminator_variables)

    initial_operation = tf.global_variables_initializer()
    session.run(initial_operation)

    # debug part
    """
    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.train.Saver()
        saver.restore(session, './backup/latest')
    """

    image_train, image_test = load()
    print(image_train.shape)
    image_train = np.array([a / 127.5 - 1 for a in image_train])  # 255/2 -> [0,1]
    print(image_train[0])
    image_test = np.array([a / 127.5 - 1 for a in image_test])

    step_num = int(len(image_train) / BATCH_SIZE)

    while True:
        session.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(session.run(epoch)))

        np.random.shuffle(image_train)

        # Completion
        if session.run(epoch) <= PRETRAIN_EPOCH:
            generator_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                image_batch = image_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()

                _, generator_loss = session.run([generator_train_optimizer, model.generator_loss],
                                                feed_dict={original_input: image_batch,
                                                           mask: mask_batch, is_training: True})
                generator_loss_value += generator_loss

            print('Completion loss: {}'.format(generator_loss_value))

            np.random.shuffle(image_test)
            image_batch = image_test[: BATCH_SIZE]
            mask_batch = get_points()
            completion = session.run(model.completion, feed_dict={original_input: image_batch,
                                                                  mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/().jpg'.format("{0:06d}".format(session.run(epoch))),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(session, './backup/latest', write_meta_graph=False)
            if session.run(epoch) == PRETRAIN_EPOCH:
                saver.save(session, './backup/pretrained', write_meta_graph=False)


        # Discrimination
        else:
            generator_loss_value = 0
            discriminator_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                image_batch = image_train[1 * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()

                _, generator_loss, completion = session.run(
                    [generator_train_optimizer, model.generator_loss, model.completion],
                    feed_dict={original_input: image_batch, mask: mask_batch, is_training: True})
                generator_loss_value += generator_loss

                local_image_batch = []
                local_completion_batch = []

                # piece, stitching, completion
                for i in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[i]  # hole coordinate
                    local_image_batch.append(image_batch[i][y1:y2, x1:x2 :])
                    local_completion_batch.append(completion[i][y1:y2, x1:x2, :])

                local_image_batch = np.array(local_image_batch)
                local_completion_batch = np.array(local_completion_batch)

                _, discriminator_loss = session.run([discriminator_train_optimizer, model.discriminator_loss],
                                                    feed_dict={original_input: image_batch, mask: mask_batch,
                                                               local_input: local_image_batch,
                                                               global_completion:completion,
                                                               local_completion: local_completion, is_training: True})
                discriminator_loss_value += discriminator_loss

            print('Completion loss: {}'.format(generator_loss_value))
            print('Discriminator loss: {}'.format(discriminator_loss_value))

            np.random.shuffle(image_test)
            image_batch = image_test[:BATCH_SIZE]
            mask_batch = get_points()
            completion = session.run(model.completion, feed_dict={original_input: image_batch,
                                                                  mask: mask_batch, is_training:False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(session.run(epoch))),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(session, './backup/latest', write_meta_graph=False)

def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):  # get pixel coordinates, set the area
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        width, height = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - width)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - height)
        p2 = p1 + width
        q2 = q1 + height

        new_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        new_mask[q1:q2 + 1, p1 + p2 + 1] =1
        mask.append(new_mask)

    return np.array(points), np.array(mask)


if __name__ == '__main__':
    train_images()

