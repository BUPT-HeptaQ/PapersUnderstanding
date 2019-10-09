import os
import time
import math
from os.path import join, exists

import cv2
import numpy as np
import h5py

TRAIN = './desktop/deep_landmark/cnn-face-data'
OUTPUT = './desktop/deep_landmark/mytrain'

if not exists(OUTPUT):
    os.mkdir(OUTPUT)
assert (exists(TRAIN) and exists(OUTPUT))


""" function part """


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


def show_landmark(face, landmark):
    """
        view face with landmark for visualization
    """
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (0,0,0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                           rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
    return face, landmark_


def flip(face, landmark):
    """
    flip the face and exchange the eyes and mouths point
    :param face: output array of the same size and type as src
    :param landmark: change landmark point
    :return:
    """

    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]

    return face_flipped_by_x, landmark_


def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def logger(msg):
    """
    log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))


def getPatch(img, bbox, point, padding):
    """
    Get a patch iamge around the given point in bbox with padding
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


def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass


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


def generate_hdf5(ftxt, output, fname, argument=False):
    data = getDataFromTxt(ftxt)
    # Divide the data into three parts. This step sets a single global variable will not affect the result.
    F_images = []  # full face
    F_landmarks = []
    EN_images = []  # eyes and nose
    EN_landmarks = []
    NM_images = []  # nose and mouth
    NM_landmarks = []

    for (imgPath, bbox, landmarkGT) in data:
        # here according to the performance of computer can choose CV_LOAD_IMAGE_GRAYSCALE or CV_LOAD_IMAGE_UNCHANGED
        """
        Specific color type of the loaded image:
        CV_LOAD_IMAGE_COLOR the loaded image is forced to be a 3-channel color image
        CV_LOAD_IMAGE_GRAYSCALE the loaded image is forced to be grayscale
        CV_LOAD_IMAGE_UNCHANGED the loaded image will be loaded as is.
        """

        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_COLOR)
        assert(img is not None)
        logger("Process %s" % imgPath)

        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)  # adjust parameters for more outputs
        f_face = img[f_bbox.top:f_bbox.bpttom+1, f_bbox.left:f_bbox.right+1]  # Face frame image

        """For the training set, the data set has been augmented. 
        In addition to mirroring, two sets of scaling and four sets of panning are performed on the face frame position.
        """

        """ data argument part """

        # data argument
        if argument and np.random.rand() > -1:   # when argument=false, do not data argument
            # flip faces
            # Mirror both the face frame image and the key coordinates
            face_flipped, landmark_flipped = flip(f_face, landmarkGT)
            # Face frame image is scaled to uniform size, default bilinear interpolation
            face_flipped = cv2.resize(face_flipped, (39, 39))
            # The image shape read by opencv is (h, w, c), now converted to (c, h, w)
            F_images.append(face_flipped.reshape((1, 39, 39)))
            # Reshape a 5x2 label into a one-dimensional
            F_landmarks.append(landmark_flipped.reshape(10))

            """
            # rotation
            # Two sets of random small-angle rotations were performed on the face frame, 
            # but the accuracy of the feature point position prediction did not improve much.
            
            if np.random.rand() > 0.5:
                # Use relative coordinates to promote model convergence and avoid divergence during network training
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox,
                                                                 bbox.reprojectLandmark(landmarkGT), 5)
                # When doing data augmentation, the corresponding 
                # feature point coordinates are converted into relative coordinates.
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (39, 39))
                F_images.append(face_rotated_by_alpha.reshape((1, 39, 39)))
                F_landmarks.append(landmark_rotated.reshape(10))

                # flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (39, 39))
                F_images.append(face_flipped.reshape((1, 39, 39)))
                F_landmarks.append(landmark_flipped.reshape(10))
            # rotation
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox,
                                                                         bbox.reprojectLandmark(landmarkGT), -5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (39, 39))
                F_images.append(face_rotated_by_alpha.reshape((1, 39, 39)))
                F_landmarks.append(landmark_rotated.reshape(10))

                # flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (39, 39))
                F_images.append(face_flipped.reshape((1, 39, 39)))
                F_landmarks.append(landmark_flipped.reshape(10))
            """
        f_face = cv2.resize(f_face, (39, 39))
        en_face = f_face[:31, :]
        nm_face = f_face[8:, :]

        f_face = f_face.reshape((1, 39, 39))
        f_landmark = landmarkGT.reshape(10)
        F_images.append(f_face)
        F_landmarks.append(f_landmark)

        # EN
        # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
        # en_face = img[en_bbox.top:en_bbox.bottom+1, en_bbox.left:en_bbox.right+1]

        # data argument
        if argument and np.random.rand() > 0.5:
            # flip faces
            face_flipped, landmark_flipped = flip(en_face, landmarkGT)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[:3, :].reshape(6)
            EN_images.append(face_flipped)
            EN_landmarks.append(landmark_flipped)

        en_face = cv2.resize(en_face, (31, 39)).reshape((1, 31, 39))
        en_landmark = landmarkGT[:3, :].reshape(6)
        EN_images.append(en_face)
        EN_landmarks.append(en_landmark)

        # NM
        # nm_bbox = bbox.subBBox(-0.05, 1.05, 0.18, 1.05)
        # nm_face = img[nm_bbox.top:nm_bbox.bottom+1, nm_bbox.left:nm_bbox.right+1]

        # data argument
        if argument and np.random.rand() > 0.5:
            # flip faces
            face_flipped, landmark_flipped = flip(nm_face, landmarkGT)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[2:, :].reshape(6)
            NM_images.append(face_flipped)
            NM_landmarks.append(landmark_flipped)

        nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 31, 39))
        nm_landmark = landmarkGT[2:, :].reshape(6)
        NM_images.append(nm_face)
        NM_landmarks.append(nm_landmark)

    # images, landmarks = process_images(ftxt, output)

    F_images, F_landmarks = np.asarray(F_images), np.asarray(F_landmarks)
    EN_images, EN_landmarks = np.asarray(EN_images), np.asarray(EN_landmarks)
    NM_images, NM_landmarks = np.asarray(NM_images), np.asarray(NM_landmarks)

    F_images = processImage(F_images)  # Image preprocessing: de-averaging, normalization
    shuffle_in_unison_scary(F_images, F_landmarks)  # shuffle
    EN_images = processImage(EN_images)
    shuffle_in_unison_scary(EN_images, EN_landmarks)
    NM_images = processImage(NM_images)
    shuffle_in_unison_scary(NM_images, NM_landmarks)

    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)  # Stitching the full path into the h5 file
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_images.astype(np.float32)
        h5['landmark'] = F_landmarks.astype(np.float32)

    # eye and nose
    base = join(OUTPUT, '1_EN')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = EN_images.astype(np.float32)  # Convert data to float32 type, save image
        h5['landmark'] = EN_landmarks.astype(np.float32)  # Convert data to float32 type, save coordinate labels

    # nose and mouth
    base = join(OUTPUT, '1_NM')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = NM_images.astype(np.float32)
        h5['landmark'] = NM_landmarks.astype(np.float32)


if __name__ == '__main__':
    # train data
    h5_path = './desktop/deep_landmark/mydataset'

    # The join function is equivalent to the fullfile function in matlab,
    # used to connect the directory and file name to get the full file path.
    train_txt = join(TRAIN, 'trainImageList.txt')
    # Input parameters: (original image and key point coordinate label text,
    # h5 file output directory, h5 file name, whether data is augmented)
    generate_hdf5(train_txt, OUTPUT, 'train.h5', argument=True)

    test_txt = join(TRAIN, 'testImageList.txt')
    # The validation set does not need to take a large number of values, no rotation
    generate_hdf5(test_txt, OUTPUT, 'test.h5')

    with open(join(OUTPUT, '1_F/train.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_f/train.h5')
    with open(join(OUTPUT, '1_EN/train.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_EN/train.h5')
    with open(join(OUTPUT, '1_NM/train.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_NM/train.h5')
    with open(join(OUTPUT, '1_F/test.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_f/test.h5')
    with open(join(OUTPUT, '1_EN/test.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_EN/test.h5')
    with open(join(OUTPUT, '1_NM/test.txt'), 'w') as fd:
        fd.write(h5_path + 'train/l_NM/test.h5')

