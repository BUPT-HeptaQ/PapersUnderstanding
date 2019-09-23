import sys
import os.path
import argparse
import caffe

import numpy as np
from scipy.misc import imread, imresize
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--caffe', help='path to caffe installation')
parser.add_argument('--model_def', help='path to model definition prototxt')
parser.add_argument('--model', help='path to model parameters')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
parser.add_argument('--image', help='path to image')
parser.add_argument('--features_save_to', help='path to image features')

args = parser.parse_args()

if args.caffe:
    caffe_path = args.caffe + '/python'
    sys.path.append(caffe_path)


def predict(in_data, net):
    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]

    return features


def batch_predict(filenames, net):
    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    all_features = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i + N, Nf))
        batch_file_names = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j, fname in enumerate(batch_file_names):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
            # RGB -> BGR
            im = im[:, :, (2, 1, 0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j, :, :, :] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        features = predict(in_data, net)

        for j in range(len(batch_range)):
            all_features[i + j, :] = features[j, :]

        print('Done %d/%d files' % (i + len(batch_range), len(filenames)))

    return all_features


if args.gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

print('start caffe')
net = caffe.Net(args.model_def, args.model, caffe.TEST)
print('load image')
all_features = batch_predict([args.image], net)
print('save mat')
scipy.io.savemat(args.features_save_to, mdict={'feats': np.transpose(allftrs)})
print('done')

