import cv2

img = cv2.imread('test.jpg')
cv2.imshow('Image', img)

# Apply Gaussian blur to the image with a kernel size of (3, 3) and standard deviation of 0
guassian = cv2.GaussianBlur(img, (3, 3), 0)

# Apply unsharp masking to enhance edges and details using a weighted combination of the original image and the Gaussian-blurred image
# The equation used is: img = a * img1 + b * img2 + y
unsharp_image = cv2.addWeighted(img, 2, guassian, -1, 0)

# Display the image after applying Gaussian blur in a window titled 'Gaussian Blur'
cv2.imshow("Gaussian Blur", guassian)

# Display the unsharpened image after applying the unsharp masking technique in a window titled 'Unsharpened Image'
cv2.imshow("sharpened Image", unsharp_image)
cv2.waitKey(0)

