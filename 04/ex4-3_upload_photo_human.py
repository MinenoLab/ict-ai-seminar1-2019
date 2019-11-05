import cv2
from datetime import datetime
import requests


# 投稿先の情報（Slack Token ID）を定義する
TOKEN_ID = "xoxb-782296539605-782306143173-XcdM9SHmucqw7dSzIMyBl5Oc"
CHANNEL_ID = "CNMF49URG"

# 投稿者の名前設定を定義する
YOUR_NAME = "Your name: XXXXX"

# カメラとプログラムを連携する
# ※ 今回は0番目
DEVICE_ID = 0
camera_device = cv2.VideoCapture(DEVICE_ID)

# カメラから画像を取得する
ret, image_data = camera_device.read()

# 保存する画像ファイルの名前を定義する
## 現在時刻の文字列を取得する
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
## 保存する画像ファイルの名前を現在時刻にする
photo_name = current_time
## 保存するパスと画像ファイル名を結合する
photo_name = "/home/pi/ictai/04/exercise4-3_save_photo/"+photo_name+".jpg"

# 画像を保存する
cv2.imwrite(photo_name, image_data)

# Slackに投稿する画像を読み込む
send_image_data = open(photo_name, "rb")
upload_data = {"file": send_image_data}

# 投稿時の設定情報
setting = {"token": TOKEN_ID,"channels": CHANNEL_ID, "title": YOUR_NAME+"/"+current_time, "filetype": "jpg", "initial_comment": "Taken by "+current_time+"."}
# Slackに対して撮影した画像を送信する
response = requests.post(url="https://slack.com/api/files.upload", params=setting, files=upload_data)

# アップロードした画像情報を画面に出力する
print("Upload photo: "+photo_name)

# カメラとプログラムの連携を解放する
camera_device.release()
cv2.destroyAllWindows()

