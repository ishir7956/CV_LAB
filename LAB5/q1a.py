import cv2
import numpy as np

def harris_corner_detector(image, k=0.04, threshold=0.01):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Calculate gradients using Sobel operators
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute elements of the Harris matrix
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Apply Gaussian smoothing to the computed elements
    kernel_size = 5
    Ix2 = cv2.GaussianBlur(Ix2, (kernel_size, kernel_size), 0)
    Iy2 = cv2.GaussianBlur(Iy2, (kernel_size, kernel_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 0)

    # Calculate the Harris response
    det_M = (Ix2 * Iy2) - (Ixy ** 2)
    trace_M = Ix2 + Iy2
    harris_response = det_M - k * (trace_M ** 2)

    # Apply thresholding to find corners
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = [0, 0, 255]  # Mark corners in red

    return corners

if __name__ == "__main__":
    # Load an image
    image = cv2.imread("C:\\210962062\\LAB5\\Resources\\chess-board-background-design_36244-122.jpg")

    # Apply the Harris Corner Detector
    corners = harris_corner_detector(image)

    # Display the original image with detected corners
    cv2.imshow("Harris Corners", cv2.addWeighted(image, 1, corners, 0.5, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
