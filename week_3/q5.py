import cv2
import numpy as np

# Load the image from the file
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(img, threshold1=100, threshold2=200)

# Display the original image and the detected edges
cv2.imshow("Original Image", img)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

