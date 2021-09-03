import cv2 as cv
import numpy as np
import os


path = os.path.dirname(os.path.realpath(__file__)) + "/DSCF0077.JPG"

img = cv.imread(path)
img = img[:1200, :1200]

# 4 matching pixel locations
x1 = [36, 413-20]; x2 = [1099, 530-20]; x3 = [1099, 706-20]; x4 = [25, 674-20]
xt1 = [25, 413-20]; xt2 = [1099, 413-20]; xt3 = [1099, 706-20]; xt4 = [25, 706-20]

# x1 = [628, 582]; x2 = [743, 589]; x3 = [741, 679]; x4 = [627, 676]
# xt1 = [627, 585]; xt2 = [735, 585]; xt3 = [735, 677]; xt4 = [627, 677]

# x1 = [54, 542]; x2 = [114, 546]; x3 = [110, 665]; x4 = [48, 663]
# xt1 = [51, 543]; xt2 = [112, 543]; xt3 = [112, 664]; xt4 = [51, 664]


# img[x1[1]-5:x1[1]+5, x1[0]-5:x1[0]+5] = [0, 0, 255]
# img[x2[1]-5:x2[1]+5, x2[0]-5:x2[0]+5] = [0, 0, 255]
# img[x3[1]-5:x3[1]+5, x3[0]-5:x3[0]+5] = [0, 0, 255]
# img[x4[1]-5:x4[1]+5, x4[0]-5:x4[0]+5] = [0, 0, 255]


srcpts = np.array([x1, x2, x3, x4], dtype=np.float32)
dstpts = np.array([xt1, xt2, xt3, xt4], dtype=np.float32)

# get homography matrix using 4 pixels that match on each picture
H = cv.getPerspectiveTransform(srcpts, dstpts, solveMethod=cv.DECOMP_LU)

# warp using H matrix
new_img = cv.warpPerspective(img, H, (2700, 1200))

new_img[0:, 1400:2600] = img

new_img = new_img[:]

cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', new_img)
cv.waitKey(0)
cv.destroyAllWindows()
