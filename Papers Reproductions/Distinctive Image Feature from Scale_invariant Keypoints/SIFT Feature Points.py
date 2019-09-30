import numpy as np
import cv2


img = cv2.imread('test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find keypoints
kp = sift.detect(gray, None)

# draw pictures for keypoints
ret = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('ret', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()

# use keypoint to find the SIFT feature vector
kp, des = sift.compute(gray, kp)

print(np.shape(kp))
print(np.shape(des))

print(des[0])
