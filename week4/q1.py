import cv2 as cv
img = cv.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\210962062\\210962062_week4\\resources\\fox.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow('fox', img)
#Simple Thresholding
threshold1, thres = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholding Image', thres)
threshold2, thres_inv = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Inverted Thresholding Image', thres_inv)
threshold3, thres_trunc = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)
cv.imshow('Truncated Thresholding', thres_trunc)
threshold4, thres_tozero = cv.threshold(img, 100, 255, cv.THRESH_TOZERO)
cv.imshow('Tozero Thresholding', thres_tozero)
threshold5, thres_tozero_inv = cv.threshold(img, 100, 255, cv.THRESH_TOZERO_INV)
cv.imshow('Tozero inverted', thres_tozero_inv)
#Adaptive Thresholding
thres_adapting = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Thresholding', thres_adapting)
#Otsu Thresholding
threshold6, thres_otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('Otsu Thresholding', thres_otsu)
cv.waitKey(0)