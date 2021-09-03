import cv2 as cv
import numpy as np
import os


path = os.path.dirname(os.path.realpath(__file__)) + "/DSCF0071.JPG"

img = cv.imread(path)


x1 = [784, 1181]; x2 = [791, 1110]; x3 = [841, 1110]; x4 = [840, 1182]
xt1 = [784, 1181]; xt2 = [784, 1111]; xt3 = [840, 1111]; xt4 = [840, 1181]

# create red rectangles to show used pixels
img[x1[1]-2:x1[1]+2, x1[0]-2:x1[0]+2] = [0, 0, 255]
img[x2[1]-2:x2[1]+2, x2[0]-2:x2[0]+2] = [0, 0, 255]
img[x3[1]-2:x3[1]+2, x3[0]-2:x3[0]+2] = [0, 0, 255]
img[x4[1]-2:x4[1]+2, x4[0]-2:x4[0]+2] = [0, 0, 255]


srcpts = np.array([x1, x2, x3, x4], dtype=np.float32)
dstpts = np.array([xt1, xt2, xt3, xt4], dtype=np.float32)


H = cv.getPerspectiveTransform(srcpts, dstpts, solveMethod=cv.DECOMP_LU)


new_img = cv.warpPerspective(img, H, (1600, 1200))


cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', new_img)
cv.waitKey(0)
cv.destroyAllWindows()
