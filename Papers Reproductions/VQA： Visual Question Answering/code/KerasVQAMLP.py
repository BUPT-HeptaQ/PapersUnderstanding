import sys
from random import shuffle
import argparse


import numpy as np
import scipy.io
import operator

from collections import defaultdict
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.utils.vis_utils import plot_model
from IPython.display import Image
from itertools import zip_longest

from sklearn import preprocessing
from sklearn.externals import joblib

import spacy

""" initial the original data """
questions_train = open('../questions_train2014.txt', 'r').read().splitlines()
answers_train = open('../answers_train2014_modal.txt', 'r').read().splitlines()
images_train = open('../images_train2014.txt', 'r').read().splitlines()

""" process the responses """

# open AI question -> multi-class classification question
# set the default the max answers can get
max_answers = 1000
answer_frequency = defaultdict(int)

# build a new dictionary 'dict' for all answers
for answer in answers_train:
    answer_frequency[answer] += 1

# according to the arise times to order them
sorted_frequency = sorted(answer_frequency.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
top_answers, top_frequency = zip(*sorted_frequency)
new_answers_train = []
new_questions_train = []
new_images_train = []

# only extract the related top 1000 responses
for answer, question, image in zip(answers_train, questions_train, images_train):
    if answer in top_answers:
        new_answers_train.append(answer)
        new_questions_train.append(question)
        new_images_train.append(image)

# refresh data set
questions_train = new_questions_train
answers_train = new_answers_train
images_train = new_images_train

# use preprocessing function in sklearn to value the label of 1000 answers
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(answers_train)
num_classes = len(list(label_encoder.classes_))
joblib.dump(label_encoder, '../label_encoder.pkl')


# this method turn all answers into digital label
def get_answers_matrix(answers, encoder):
    # string turn into digital expression
    encoded_answers = encoder.transform(answers)
    encoded_classes = encoder.classes_.shape[0]
    # made the standard matrix
    encoded_matrix = np_utils.to_categorical(encoded_answers, encoded_classes)

    return encoded_matrix


""" process the input images """

# imported downloaded vgg_features (the scipy can read matlab file)
vgg_model_path = '../vgg_feats.mat'

# load VGG features
features_structure = scipy.io.loadmat(vgg_model_path)
VGG_features = features_structure['feats']

# Match the picture one by one
image_ids = open('../coco_vgg_IDMap.txt').read().splitlines()
id_map = {}

for ids in image_ids:
    id_split = ids.split()
    id_map[id_split[0]] = int(id_split[1])


# the matrix is the digital expression of the input images
# The matrix is the 4096-dimensional array obtained by adding flatten layer
# after the image is scanned/processed by VGG, which is called CNN Features
def get_images_matrix(img_coco_ids, img_map, VGG_features):
    img_samples = len(img_coco_ids)
    img_dimensions = VGG_features.shape[0]
    image_matrix = np.zeros((img_samples, img_dimensions))

    for i in range(len(img_coco_ids)):
        image_matrix[i, :] = VGG_features[:, img_map[img_coco_ids[i]]]

    return image_matrix


""" process the text questions"""

# use SpaCy transfer all questions to vectors, and averaged all sentences, like Word2Vec
# load the English library in SpaCy
nlp = spacy.load('en')
# The dimensional size of the images
img_dimension = 4096
# Sentence/word dimension size
word_vec_dimension = 300


# This method is used to calculate the sum of all word vectors in the sentence.
# The purpose is to represent our text with Numbers
def get_questions_matrix_sum(questions, nlp):
    # assert not isinstance(questions, basestring)
    question_samples = len(questions)
    word_vec_dimension = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((question_samples, word_vec_dimension))

    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            questions_matrix[i, :] += tokens[j].vector

    return questions_matrix


""" VQA model: MLP"""
# parameters
num_hidden_units = 1024
num_hidden_layers = 3
dropout = 0.5
activation = 'tanh'
num_epochs = 50
model_save_interval = 10
batch_size = 128

# build MLP
# input layer
model = Sequential()
model.add(Dense(num_hidden_units, input_dim=img_dimension+word_vec_dimension, kernel_initializer='uniform'))
model.add(Activation(activation))  # activation avoid over fitting
model.add(Dropout(dropout))

# meddle layer
for i in range(num_hidden_layers-1):
    model.add(Dense(num_hidden_units, kernel_initializer='uniform'))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

# output layer
model.add(Dense(num_classes, kernel_initializer='uniform'))
model.add(Activation('softmax'))

# print the model
plot_model(model, to_file='../model_mlp.png', show_shapes=True)
Image(filename='../model_mlp.png')

# save the printed model
json_string = model.to_json()
model_file_name = '../mlp_num_hidden_units_' + str(num_hidden_units) + '_num_hidden_layers_' + str(num_hidden_layers)
open(model_file_name + '.json', 'w').write(json_string)


# this is a standard chunk list method, separate original data by batches, prepare for the batch training
#  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
def grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n

    return zip_longest(*args, fillvalue=fill_value)


# compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# training data
for k in range(num_epochs):
    # shuffle the data
    index_shuffle = [i for i in range(len(questions_train))]
    shuffle(index_shuffle)

    # get questions, answers and images one by one
    questions_train = [questions_train[i] for i in index_shuffle]
    answers_train = [answers_train[i] for i in index_shuffle]
    images_train = [images_train[i] for i in index_shuffle]

    # this bar shows the training state, if need do some customize could use this
    progress_bar = generic_utils.Progbar(len(questions_train))

    # batch data
    for question_batch, answer_batch, image_batch in zip(
            grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
            grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
            grouper(images_train, batch_size, fillvalue=images_train[-1])):

        input_question_batch = get_questions_matrix_sum(question_batch, nlp)
        input_image_batch = get_images_matrix(image_batch, id_map, VGG_features)

        output_batch = get_answers_matrix(answer_batch, label_encoder)
        loss = model.train_on_batch([input_question_batch, input_image_batch], output_batch)
        progress_bar.add(batch_size, values=[("train loss", loss)])

    # model_save_interval is 10, which means every 10 times save the model
    if k % model_save_interval == 0:
        model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

# save the final model
model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

