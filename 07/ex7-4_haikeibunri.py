import cv2
import numpy as np

img1 = cv2.imread("background.jpg")
img1 = cv2.GaussianBlur(img1, (5, 5), 0)

img2 = cv2.imread("background_withcat.jpg")
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = fgbg.apply(np.uint8(img1))
fgmask = fgbg.apply(np.uint8(img2))

cv2.imwrite("result4_background_mask.jpg", fgmask)