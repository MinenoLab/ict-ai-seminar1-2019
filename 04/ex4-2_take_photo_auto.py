import cv2
from datetime import datetime
import time


# カメラとプログラムを連携する
# ※ 今回は0番目
DEVICE_ID = 0
camera_device = cv2.VideoCapture(DEVICE_ID)

# 撮影する画像枚数の設定
TAKE_NUM = 3
# 撮影して，保存した画像をカウントする変数
counter = 0
# 画像を撮影し，保存する処理を繰り返す
while counter < TAKE_NUM:
    # 何枚目の写真撮影処理かを画面に出力する
    print("Take photo "+str(counter+1)+"/"+str(TAKE_NUM)+" times")
    # カメラから画像を取得する
    ret, image_data = camera_device.read()
    
    # 保存する画像ファイルの名前を定義する
    ## 現在時刻の文字列を取得する
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    ## 保存する画像ファイルの名前を現在時刻にする
    photo_name = current_time
    ## 保存するパスと画像ファイル名を結合する
    photo_name = "/home/pi/ictai/04/exercise4-2_save_photo/"+photo_name+".jpg"

    # 画像を保存する
    cv2.imwrite(photo_name, image_data)

    # 保存した画像情報を画面に出力する
    print("Save "+photo_name+".")

    # 写真を保存後，3秒間待つ
    time.sleep(3)

    # カウンター（撮影した枚数）を増やす
    counter = counter + 1

# カメラとプログラムの連携を解放する
camera_device.release()
cv2.destroyAllWindows()

