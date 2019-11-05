import cv2


# カメラとプログラムを連携する
# ※ 今回は0番目を指定する
DEVICE_ID = 0
camera_device = cv2.VideoCapture(DEVICE_ID)

# カメラから画像を取得する
ret, image_data = camera_device.read()

# 保存する画像ファイルの名前を定義する
photo_name = "default"
# 保存するパスと画像ファイル名を結合する
photo_name = "/home/pi/ictai/04/exercise4-1_save_photo/"+photo_name+".jpg"

# 画像を保存する
cv2.imwrite(photo_name, image_data)

# 保存した画像情報を画面に出力する
print("Save "+photo_name+".")

# カメラとプログラムの連携を解放する
camera_device.release()
cv2.destroyAllWindows()

