import cv2

# Load the image from the file
img = cv2.imread('test.jpg')

# Apply box filter to the image
box_filtered = cv2.boxFilter(img, -1, (3, 3))

# Apply Gaussian filter to the image
gaussian_filtered = cv2.GaussianBlur(img, (3, 3), 0)

# Display the original image, box filtered image, and Gaussian filtered image
cv2.imshow("Original Image", img)
cv2.imshow("Box Filtered Image", box_filtered)
cv2.imshow("Gaussian Filtered Image", gaussian_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
