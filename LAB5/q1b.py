import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('C:\\210962062\\LAB5\\Resources\\chess-board-background-design_36244-122.jpg', cv.IMREAD_GRAYSCALE)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints


kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("non maxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with non maxSuppression: {}".format(len(kp)))

cv.imshow('fast_true', img2)

fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without non maxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imshow('fast_false', img3)

cv.waitKey()