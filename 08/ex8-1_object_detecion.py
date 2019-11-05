import cv2
import time
import os
import copy
import numpy as np
from sklearn.cluster import KMeans

class detection: 
    # 初期化関数 
    def __init__(self):        
        # 解像度の設定 初期値 camera_width = 400, camera_height = 300
        self.camera_width = 400
        self.camera_height = 300

        # 変数初期化
        self.fps = ""
        self.vidfps = 20
        self.elapsedTime = 0
        self.message = "Push [p] to take a background picture."
        self.flag_detection = False

        # カメラの設定
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.vidfps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(1)

    # 背景分離関数
    def MOG_map(self, img1, img2):
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
        fgmask = fgbg.apply(np.uint8(img1))
        fgmask = fgbg.apply(np.uint8(img2))
        cv2.imshow("fgmask", fgmask)
        return fgmask

    # バウンディングボックス描画用のx,y座標取得関数
    def get_x_y_limit(self, Y, X):
        x_max = np.max(X)
        x_min = np.min(X)
        y_max = np.max(Y)
        y_min = np.min(Y)
        return x_min, y_min, x_max, y_max

    # バウンディングボックスの描画関数
    def bounding_box(self, img, x_min, y_min, x_max, y_max):
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
        return img

    def main(self):
        while self.cap.isOpened():
            t1 = time.time()

            # カメラからの画像を取得
            ret, frame = self.cap.read()
            if not ret:
                break

            # キーボードからの入力を取得
            key = cv2.waitKey(1)&0xFF

            # pキーで検出開始
            if key == ord('p'):
                # 背景分離で使用する背景画像を保存
                background_img = copy.deepcopy(frame)
                self.message = "Start object detection."
                self.flag_detection = True

            # qキーで検出終了
            if key == ord('q'):
                break
            
            if self.flag_detection == True:
                # 背景分離処理
                img = self.MOG_map(frame, background_img)
                
                if np.max(img) > 200:
                    # 物体検出
                    Y, X = np.where(img > 200)
                    
                    # バウンディングボックスを描画
                    x_min, y_min, x_max, y_max = self.get_x_y_limit(Y, X)
                    frame = self.bounding_box(frame, x_min, y_min, x_max, y_max)

            # message
            cv2.putText(frame, self.fps, (self.camera_width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 0 ,0), 1, cv2.LINE_AA)

            # 画像の描画
            cv2.imshow("Result", frame)

            # FPSの算出
            elapsedTime = time.time() - t1
            self.fps = "{:.0f} FPS".format(1/elapsedTime)
            
        # すべての描画ウィンドウを閉じる
        cv2.destroyAllWindows()

main = detection()
main.main()
