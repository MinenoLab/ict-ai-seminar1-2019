import cv2

img = cv2.imread("cat_original.jpg")

img_result = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite("result1_rotate.jpg", img_result)