import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

showSteps = False
dirpath = os.path.dirname(os.path.realpath(__file__))

def useFile(file):
    if file.endswith(".JPG"):
        num = int(file[-5])
        if num % 2 == 0:
            return True

images = [cv.imread(dirpath + "/pics/" + file) for file in sorted(os.listdir(dirpath + "/pics/")) if useFile(file)]

def showImg(img, finalStep=False, name="img"):
    if showSteps or finalStep:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()


stitcher = cv.Stitcher.create(mode=cv.STITCHER_PANORAMA)
res = stitcher.stitch(images)

cv.imwrite("result.jpg", res[-1])
# showImg(res[-1], True)
