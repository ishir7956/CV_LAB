import cv2
import numpy as np

# Load the image from the file
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate gradients using the Sobel operator
gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of gradients
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Calculate the direction (angle) of gradients
gradient_direction = np.arctan2(gradient_y, gradient_x)

# Normalize the magnitude to the range [0, 255] for display
gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the original image, gradient magnitude, and gradient direction
cv2.imshow("Original Image", img)
cv2.imshow("Gradient Magnitude", gradient_magnitude_normalized)
cv2.imshow("Gradient Direction", gradient_direction)
cv2.waitKey(0)
cv2.destroyAllWindows()
