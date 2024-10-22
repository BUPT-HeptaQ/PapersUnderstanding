import os
import cv2
import sys
caffe_root = 'desktop/caffe/'
sys.path.insert(0, caffe_root + 'python')

import numpy as np
from os.path import join
import caffe


class CNN(object):
    """
    Generalized CNN for simple run forward with given Model
    """
    def __init__(self, net, model):
        self.net = net
        self.model = model
        self.cnn = caffe.Net(net, model, caffe.TEST)  # fails if not exists

    def forward(self, data, layer='fc2'):
        print(data.shape)
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(int(len(x)/2))])
        result = t(result)
        return result


class BBox(object):
    """Bounding Box of face """
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
        # print(len(landmark))
        if not len(landmark) == 5:
            landmark = landmark[0]
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

    def cropImage(self, image):
        """
        crop images with left, right, top, bottom  **Make sure is not out of box**
        """
        return image[self.top:self.bottom+1, self.left:self.right+1]


class Landmarker(object):
    """
    class Landmarker wrapper functions for predicting facial landmarks
    """
    def __init__(self):
        """
        initialize Landmarker with files under VERSION
        """
        # model_path = join(PROJECT_ROOT, VERSION)
        deploy_path = './desktop/deep_landmark/prototxt'
        model_path = './desktop/deep_landmark/model'
        CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']
        level1 = [(join(deploy_path, '1_F_deploy.prototxt'), join(model_path, '1_F/_iter_10000.caffemodel'))]
        level2 = [(join(deploy_path, '2_%s_deploy.prototxt' % name), join(model_path, '2_%s/_iter_10000.caffemodel'
                                                                          % name))for name in CNN_TYPES]
        level3 = [(join(deploy_path, '3_%s_deploy.prototxt' % name), join(model_path, '3_%s/_iter_10000.caffemodel'
                                                                          % name)) for name in CNN_TYPES]
        self.level1 = [CNN(deploy, model) for deploy, model in level1]
        self.level2 = [CNN(deploy, model) for deploy, model in level2]
        self.level3 = [CNN(deploy, model) for deploy, model in level3]

    def _processImage(self, images):
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

    def detectLandmark(self, image, bbox, mode='three'):
        """
        Predict landmarks for face with bbox in image fast mode will only apply level-1 and level-2
        """
        # if not isinstance(bbox, BBox) or image is None:
        # return None, False
        face = bbox.cropImage(image)
        # face = image
        # print(face.shape)

        face = cv2.resize(face, (39, 39))
        # print(face.shape)
        face = face.reshape((1, 1, 39, 39))
        face = self._processImage(face)

        # level-1, only F in implemented
        landmark = self.level1[0].forward(face)  # forward broadcasting
        # level-2

        landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
        if mode == 'fast':
            return landmark, True
        landmark = self._level(image, bbox, landmark, self.level3, [0.11, 0.12])

        return landmark

    def _level(self, image, bbox, landmark, cnns, padding):
        """ level-?"""
        for i in range(5):
            x, y = landmark[i]
            patch, patch_bbox = self._getPatch(image, bbox, (x, y), padding[0])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d1 = cnns[i].forward(patch)  # size = 1x2

            patch, patch_bbox = self._getPatch(image, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i + 5].forward(patch)
            # position covert
            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, image, bbox, point, padding):
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

        patch = image[int(round(patch_top)): int(round(patch_bottom + 1)),
                      int(round(patch_left)): int(round(patch_right + 1))]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])

        return patch, patch_bbox


def drawLandmark(img,  landmark):
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    return img


if __name__ == '__main__':
    result_path = './desktop/deep_landmark/result-folder'
    test_folder = './desktop/deep_landmark/test-folder'
    test_images = os.listdir(test_folder)

    for image in test_images:
        images = cv2.imread(test_folder+image)
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        cv2.resize(gray, (256, 256))

        bbox = BBox([70, 190, 70, 200])
        cv2.rectangle(images, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)

        get_landmark = Landmarker()
        final_landmark = get_landmark.detectLandmark(gray, bbox)
        print(final_landmark)

        final_landmark = bbox.reprojectLandmark(final_landmark)
        print(final_landmark)
        print(final_landmark.shape)

        img = drawLandmark(image, final_landmark)
        cv2.imwrite(result_path+'level1-'+image+'level2-'+'level3.jpg', img)

