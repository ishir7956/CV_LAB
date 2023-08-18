import cv2
img = cv2.imread('test.jpg')
new = cv2.resize(img, (500,200))
crop = img[0:500, 0:500]
cv2.imshow('original', img)
cv2.imshow('resize', new)
cv2.imshow('cropped', crop)
cv2.waitKey(0)
