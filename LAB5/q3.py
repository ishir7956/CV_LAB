import cv2
import numpy as np

# Load the reference HOG descriptor
reference_hog = np.load('reference_hog.npy')

# Load the test image
test_image = cv2.imread('C:\\210962062\\LAB5\\Resources\\solo.jpg', cv2.IMREAD_GRAYSCALE)

# Define the sliding window parameters
window_size = (64, 128)  # Width x Height
stride = 16  # Stride for moving the window

# Initialize lists to store detected windows and scores
detected_windows = []
scores = []

# Loop over different scales
for scale in [0.8, 1.0, 1.2]:
    # Resize the image
    resized_image = cv2.resize(test_image, None, fx=scale, fy=scale)

    # Loop over the image using a sliding window
    for y in range(0, resized_image.shape[0] - window_size[1], stride):
        for x in range(0, resized_image.shape[1] - window_size[0], stride):
            # Extract the window
            window = resized_image[y:y + window_size[1], x:x + window_size[0]]

            # Compute HOG features for the window
            hog = cv2.HOGDescriptor()
            window_hog = hog.compute(window)

            # Calculate similarity score (e.g., cosine similarity)
            similarity_score = np.dot(reference_hog.T, window_hog)[0, 0] / (np.linalg.norm(reference_hog) * np.linalg.norm(window_hog))

            # Set a threshold and collect positive detections
            if similarity_score > 0.7:
                detected_windows.append((x, y, x + window_size[0], y + window_size[1]))
                scores.append(similarity_score)

# Apply non-maximum suppression to remove overlapping detections
indices = cv2.dnn.NMSBoxes(detected_windows, scores, score_threshold=0.2, nms_threshold=0.3)

# Draw bounding boxes on the original image for the final detections
for i in indices:
    x1, y1, x2, y2 = detected_windows[i[0]]
    cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Object Detection', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
