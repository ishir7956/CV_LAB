import cv2
import numpy as np

# Load an image
image = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962062\\210962062_week4\\resources\\fox.jpg")

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_color = np.array([0, 50, 50])
upper_color = np.array([100, 255, 255])

# Create a mask based on the specified color range
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Apply the mask to the original image to get the segmented region
segmented_image = cv2.bitwise_and(image, image, mask=color_mask)

# Display the original image and the segmented image
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
