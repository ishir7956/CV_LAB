import cv2
import numpy as np


img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image',img)

lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian",lap)

cv2.waitKey(0)