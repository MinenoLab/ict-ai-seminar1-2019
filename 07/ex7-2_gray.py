import cv2

img = cv2.imread("cat_original.jpg")

img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("result2_gray.jpg", img_result)