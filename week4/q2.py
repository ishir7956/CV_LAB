import cv2
import numpy as np
import math

# Load an image
image = cv2.imread('C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962062\\210962062_week4\\resources\\sudoku.png', cv2.IMREAD_GRAYSCALE)

# Apply canny edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Apply Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

# Draw the detected lines on the original image
for i in range(0, len(lines)):
  rho = lines[i][0][0]
  theta = lines[i][0][1]
  a = math.cos(theta)
  b = math.sin(theta)
  x0 = a * rho
  y0 = b * rho
  pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
  pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

  cv2.line(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)


# Display the result
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
