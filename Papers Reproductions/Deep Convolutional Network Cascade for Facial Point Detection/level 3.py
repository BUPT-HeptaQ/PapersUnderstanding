import os
import time
import math
from os.path import join, exists
from collections import defaultdict

import cv2
import numpy as np
import h5py

""" function part """


def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def getDataFromTxt(txt, with_landmark=True):
    """
    Generate data from txt file
    return [(img_path, bbox, landmark)]
        bbox:[left, right, top, bottom]
        landmark:[(x1, y1), (x2, y2), ...]

    """
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        # according to operation system to segment these file paths
        img_path = os.path.join(dirname, components[0].replace('\\', '/'))
        # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
        bbox = [int(_) for _ in bbox]

        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))

        for index in range(0, 5):
            coordinate_value = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = coordinate_value

        for index, real_coordinate in enumerate(landmark):
            coordinate_value = ((real_coordinate[0]-bbox[0])/(bbox[1]-bbox[0]),
                                (real_coordinate[1]-bbox[2])/(bbox[3]-bbox[2]))
            landmark[index] = coordinate_value
        result.append((img_path, BBox(bbox), landmark))

    return result


def logger(msg):
    """
    log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))


def randomShift(landmarkGT, shift):
    """
    Random Shift one time
    """
    different = np.random.rand(5, 2)  # different: difference matrix
    different = (2 * different - 1) * shift
    landmarkP = landmarkGT + different
    return landmarkP


def randomShiftWithArgument(landmarkGT, shift):
    """
    Random Shift more
    """
    n = 2
    landmarkPs = np.zeros((n, 5, 2))
    for i in range(n):
        landmarkPs[i] = randomShift(landmarkGT, shift)
    return landmarkPs


def getPatch(img, bbox, point, padding):
    """
    Get a patch image around the given point in bbox with padding
    point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h

    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding

    patch = img[int(round(patch_top)): int(round(patch_bottom+1)), int(round(patch_left)): int(round(patch_right+1))]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])

    return patch, patch_bbox


def processImage(images):
    """
    process images before feeding to CNNs
    images: N x 1 x W x H
    """
    images = images.astype(np.float32)
    for i, image in enumerate(images):
        m = image.mean()
        s = image.std()
        images[i] = (image - m) / s

    return images


def shuffle_in_unison_scary(sample, label):
    rng_state = np.random.get_state()
    np.random.shuffle(sample)
    np.random.set_state(rng_state)
    np.random.shuffle(label)


"""  Bounding Box of face """


class BBox(object):

    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR

        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta

        return BBox([left, right, top, bottom])


types = [(0, 'LE1', 0.11), (0, 'LE2', 0.12), (1, 'RE1', 0.11), (1, 'RE2', 0.12), (2, 'N1', 0.11),
         (2, 'N2', 0.12), (3, 'LM1', 0.11), (3, 'LM2', 0.12), (4, 'RM1', 0.11), (4, 'RM2', 0.12)]
# 5 key points, two padding
for t in types:
    path = './desktop/deep_landmark/mytrain/3_%s' % t[1]
    createDir(path)
    # Create a folder to store the txt and h5 files for train and test


def generate(ftxt, mode, argument=False):
    """
    Generate Training Data for LEVEL-3
    mode = train or test
    """
    data = getDataFromTxt(ftxt)  # get image_path, bbox, landmark

    trainData = defaultdict(lambda: dict(patches=[], landmarks=[]))  # data dictionary
    for (imgPath, bbox, landmarkGT) in data:
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_COLOR)
        assert (img is not None)
        logger("process %s" % imgPath)

    # Do 2 sets of random translations on the relative coordinates of the key points to get 2 sets of "new key points"
        landmarkPs = randomShiftWithArgument(landmarkGT, 0.01)
        if not argument:
            landmarkPs = [landmarkPs[0]]  # The test set only does a random translation

        # For the 2 sets of random translations, the relative coordinates of all partial frame images
        # and key points relative to the partial frame are sent to the data dictionary trainData
        for landmarkP in landmarkPs:
            for idx, name, padding in types:
                # Partial frame image and partial frame according to relative coordinates
                # and padding of randomly shifted key points
                patch, patch_bbox = getPatch(img, bbox, landmarkP[idx], padding)
                # Partial frame image is scaled to 15x15
                patch = cv2.resize(patch, (15, 15))
                # Each patch is c, h, w, append becomes n, c, h, w
                patch = patch.reshape((1, 15, 15))
                trainData[name]['patches'].append(patch)
                # ‘true’ key point, then project to the partial box
                # to get the relative coordinates of the relative partial frame
                _ = patch_bbox.project(bbox.reproject(landmarkGT[idx]))
                trainData[name]['landmarks'].append(_)

    for idx, name, padding in types:
        logger('writing training data of %s' % name)
        patches = np.asarray(trainData[name]['patches'])  # get data from data dictionary
        landmarks = np.asarray(trainData[name]['landmarks'])
        patches = processImage(patches)  # Pretreatment, de-average, normalization

        shuffle_in_unison_scary(patches, landmarks)

        # generate mode.h5(train/test)
        with h5py.File('./desktop/deep_landmark/mytrain/3_%s/%s.h5' % (name, mode), 'w') as h5:
            h5['data'] = patches.astype(np.float32)
            h5['landmark'] = landmarks.astype(np.float32)

        # generate mode.txt(train/test), write into h5fil path
        with open('./desktop/deep_landmark/mytrain/3_%s/%s.txt' % (name, mode), 'w') as fd:
            fd.write('./desktop/deep_landmark/mytrain/3_%s/%s.h5' % (name, mode))


if __name__ == '__main__':
    # Seed specifies the integer value at which the algorithm used to generate the random number starts,
    # so that the random value is generated over time, and the random number is not generated every time.
    np.random.seed(int(time.time()))

    # generate train.h5和train.txt
    # The training set does data enhancement (actually just doing a random set of translations)
    # trainImageList.txt
    generate('./desktop/deep_landmark/cnn-face-data/trainImageList.txt', 'train', argument=True)
    # testImageList.txt
    generate('./desktop/deep_landmark/cnn-face-data/testImageList.txt', 'test')  # generate test.h5 and test.txt

