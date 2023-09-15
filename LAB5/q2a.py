import cv2

# Load two images
image1 = cv2.imread('C:\\210962062\\LAB5\\Resources\\image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('C:\\210962062\LAB5\\Resources\\image2.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect key points and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Match descriptors between the two images
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the matches
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow('SIFT Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

