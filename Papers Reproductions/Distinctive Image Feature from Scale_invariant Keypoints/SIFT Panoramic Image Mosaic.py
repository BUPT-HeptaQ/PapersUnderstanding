import cv2
import numpy as np
import matplotlib.pyplot as plt


class Stitcher:

    def stitch(self, imgs, ratio=0.75, reproThresh=4, showMathes=False):
        (imageB, imageA) = imgs
        # Step 1：Calculate kpsA and dpsA
        (kpsA, dpsA) = self.detectandcompute(imageA)
        (kpsB, dpsB) = self.detectandcompute(imageB)
        
        # Obtaining a matrix of changes H 
        H_matrix = self.matchKeypoint(kpsA, dpsA, kpsB, dpsB, ratio, reproThresh)

        if H_matrix is None:
            return None
        (matches, H, status) = H_matrix

        # Step 4：use cv2.warpPerspective to get an image after H change
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))

        # Step 5：Fill the image B into the image after the change of H to obtain the final image
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        if showMathes:
            # Step 6：connect key points of the image
            via = self.showMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (via, result)
        return result

        # Draw images

    def showMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Splicing two images.  constructing an all-zero matrix based on the size of the image
        via = np.zeros((max(imageB.shape[0], imageA.shape[0]), imageA.shape[1] + imageB.shape[1], 3), np.uint8)
        
        # Place image A and image B in an Image that is all zero
        via[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
        via[0:imageB.shape[0], imageA.shape[1]:] = imageB
        
        # according to the index in the maches, construct the position information of the points
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0] + imageA.shape[1]), int(kpsB[trainIdx][1]))
                # use cv2.line to draw pictures 
                cv2.line(via, ptA, ptB, (0, 255, 0), 1)

        return via

    def matchKeypoint(self, kpsA, dpsA, kpsB, dpsB, ratio, reproThresh):
        # Step 2：instantiate BFM matching and find indexes that match the added key points
        bf = cv2.BFMatcher()

        matcher = bf.knnMatch(dpsA, dpsB, 2)
        matches = []

        for match in matcher:
            if len(match) == 2 and match[0].distance < match[1].distance * ratio:
                # join match[0] matrix
                matches.append((match[0].trainIdx, match[0].queryIdx))
        
        # Step 3：use cv2.findHomography to find complianced H matrix
        if len(matches) > 4:
            # Find qualified locations based on the index
            kpsA = np.float32([kpsA[i] for (_, i) in matches])
            kpsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(kpsA, kpsB, cv2.RANSAC, reproThresh)

            return (matches, H, status)
        return None

    def cv_show(self, img, name):
        cv2.imshow(name, img)

    def detectandcompute(self, image):
        # gray value conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # instantianting a SIFT function
        sift = cv2.xfeatures2d.SIFT_create()
        # get kps key point and feature vector SIFT of dps 
        kps, dps = sift.detectAndCompute(gray, None)
        # get location information of feature points and convert data types
        kps = np.float32([kp.pt for kp in kps])

        return kps, dps
        
        
