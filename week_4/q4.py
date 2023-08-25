import numpy as np
import cv2

# Load an image
image = cv2.imread('C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962062\\210962062_week4\\resources\\fox.jpg')

# Reshape the image into a 2D array of pixels
pixels = image.reshape(-1, 3).astype(np.float32)

# Define the number of clusters (K)
num_clusters = 5

# Define criteria and apply K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
f, labels, cluster_centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert cluster centers to integers
cluster_centers = np.uint8(cluster_centers)

# Create a segmented image by assigning each pixel to its cluster's color
segmented_image = cluster_centers[labels.flatten()]

# Reshape the segmented image to the original shape
segmented_image = segmented_image.reshape(image.shape)

# Display the original image and the segmented image
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

