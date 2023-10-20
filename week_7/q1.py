import cv2
import numpy as np
import glob
checkerboard=(12,12)
criteria=(cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

objpoints=[]
imgpoints=[]

objp=np.zeros((1,checkerboard[0]*checkerboard[1],3), np.float32)
objp[0, :, :2]=np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1,2)
print(objp)
prev_img_shape=None

images=glob.glob('Resources/*.tif')

for fname in images:
    print(fname)
    img=cv2.imread(fname)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners=cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret==True:
        objpoints.append(objp)
        #corners=cv2.cornersSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        img=cv2.drawChessboardCorners(img, checkerboard, corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
h,w=img.shape[:2]

ret, camera_matrix, distortion_matrix, rvec, tvec=cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
print('Camera matrix:')
print(camera_matrix)
print('Distortion:')
print(distortion_matrix)
print('rvecs:')
print(rvec)
print('tvecs:')
print(tvec)



img=cv2.imread('resources/Image11.tif')
h,w=img.shape[:2]
new_matrix, roi =cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_matrix, (w,h) , 1, (w,h))
dst=cv2.undistort(img, camera_matrix, distortion_matrix, None, new_matrix)
x,y,w,h=roi
#dst=dst[y:y+h, w:w+h]
cv2.imshow('reprojected', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
