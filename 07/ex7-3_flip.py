import cv2

img = cv2.imread("cat_original.jpg")

img_result = cv2.flip(img, 1)

cv2.imwrite("result3_flip.jpg", img_result)