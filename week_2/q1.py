import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Path to the image
path = r'C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962062\\week_2\\test.jpg'

# Load the image in grayscale
img = cv.imread(path, cv.IMREAD_GRAYSCALE)

# Apply histogram equalization
equ = cv.equalizeHist(img)

# Concatenate the original and equalized images side by side
res = np.hstack((img, equ))

# Calculate histograms for the original and equalized images
histr = cv.calcHist([img], [0], None, [256], [0, 256])
histr1 = cv.calcHist([equ], [0], None, [256], [0, 256])

# Display the histograms using matplotlib
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(histr)
plt.title('Histogram of Original Image')
plt.subplot(1, 2, 2)
plt.plot(histr1)
plt.title('Histogram of Equalized Image')
plt.show()

# Display the concatenated image
cv.imshow('Result', res)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()

