import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_specification(input_image, reference_image):
    # Load input and reference images in grayscale
    input_img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    # Calculate histograms for the input and reference images
    hist_input = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    hist_ref = cv2.calcHist([ref_img], [0], None, [256], [0, 256])

    # Normalize histograms
    hist_input /= hist_input.sum()
    hist_ref /= hist_ref.sum()

    # Compute cumulative distribution functions (CDF) for histograms
    cdf_input = hist_input.cumsum()
    cdf_ref = hist_ref.cumsum()

    # Create a lookup table to map input image intensities to reference CDF
    lut = np.interp(cdf_input, cdf_ref, np.arange(256))

    # Apply histogram specification to the input image
    output_img = cv2.LUT(input_img, lut).astype(np.uint8)

    return output_img, hist_input, hist_ref

# Paths to input and reference images
input_path = 'input.jpg'
reference_path = 'test.jpg'

# Perform histogram specification
output_image, hist_input, hist_ref = histogram_specification(input_path, reference_path)

# Display input, reference, and histogram-specified images along with histograms
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE))
plt.title('Input Image')

plt.subplot(2, 3, 2)
plt.imshow(cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE))
plt.title('Reference Image')

plt.subplot(2, 3, 3)
plt.imshow(output_image, cmap='gray')
plt.title('Histogram Specified Image')

plt.subplot(2, 3, 4)
plt.plot(hist_input)
plt.title('Histogram of Input Image')

plt.subplot(2, 3, 5)
plt.plot(hist_ref)
plt.title('Histogram of Reference Image')

plt.subplot(2, 3, 6)
hist_output = cv2.calcHist([output_image], [0], None, [256], [0, 256])
plt.plot(hist_output)
plt.title('Histogram of Specified Image')

plt.tight_layout()
plt.show()


