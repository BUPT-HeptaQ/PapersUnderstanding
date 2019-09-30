import cv2
import numpy as np

# load in images
imgA = cv2.imread('box.png', 0)
imgB = cv2.imread('box_in_scene.png', 0)


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 1: build SIFT, find out feature points and SIFT feature vectors
sift = cv2.xfeatures2d.SIFT_create()
kpsA, dpA = sift.detectAndCompute(imgA, None)
kpsB, dpB = sift.detectAndCompute(imgB, None)

# Step 2: build BFMatcher() brute force match, Match the SIFT feature vector distance to the nearest component 
bf = cv2.BFMatcher()

# get the match output  
matches = bf.match(dpA, dpB)

# Step 3: Sort the matching results by distance
matches = sorted(matches, key=lambda x: x.distance)

# Step 4: Use cv2.drawMacthes to draw images
ret = cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches[:10], None, flags=2)

cv2.imshow('ret', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()

