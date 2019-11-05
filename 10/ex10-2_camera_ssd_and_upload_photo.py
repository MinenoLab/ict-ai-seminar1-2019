import cv2
import time
import os
import warnings
import numpy as np
from sklearn.cluster import KMeans

import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

# [10-2-1].自作のライブラリを読み込む
from ex10_upload_photo import upload_photo_to_slack

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class detection:
    # 初期化関数
    def __init__(self):
        # 解像度の設定 初期値 camera_width = 160, camera_height = 120
        self.camera_width = 160
        self.camera_height = 120

        self.input_width = self.camera_width
        self.input_height = self.camera_height

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

        np.set_printoptions(suppress=True)

        # tensorflowの初期化
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.45
        set_session(tf.Session(config=self.config))

        # クラス名を定義
        self.voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                    'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                    'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                    'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        self.class_num = len(self.voc_classes) + 1

        # 入力サイズの指定　左から高さ，横幅，チャネル数（RGBの3チャネル）
        input_shape=(self.input_height, self.input_width, 3)
        # SSDモデルを読み込み
        self.model = SSD300(input_shape, num_classes=self.class_num)
        # SSDモデルの重みへ学習済みの重みを上書き
        self.model.load_weights('weights_SSD300.hdf5', by_name=True)
        self.bbox_util = BBoxUtility(self.class_num)

        # クラスごとにバウンディングボックスの色を定義
        self.class_colors = []
        for i in range(0, self.class_num - 1):
            # This can probably be written in a more elegant manner
            hue = 255*i/(self.class_num - 1)
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col)
        
        # [10-2-2].Slackへ1度しかアップロードを行わないようにするためのアップロードフラグ
        self.is_upload = True

        time.sleep(1)

    # バウンディングボックスの描画関数
    def bounding_box(self, img, x_min, y_min, x_max, y_max, class_num):
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), self.class_colors[class_num], 5)
        return img

    def main(self):
        while self.cap.isOpened():
            t1 = time.time()

            # カメラからの画像を取得
            ret, frame = self.cap.read()

            if not ret:
                break

            key = cv2.waitKey(1)&0xFF

            # pキーで推定開始
            if key == ord('p'):
                self.message = "Start object detection."
                self.flag_detection = True

            # qキーで推定終了
            if key == ord('q'):
                #self.writer.release()##movie save
                break
            
            # [10-2-3].rキーでSlackへのアップロードフラグをリセット
            if key == ord('r'):
                self.is_upload = True

            if self.flag_detection == True:
                # カメラ画像をSSDモデルへ入力可能な形へ加工
                inputs = []
                images = []
                img = image.img_to_array(frame)
                inputs.append(img.copy())
                inputs = preprocess_input(np.array(inputs))
                
                # SSDモデルへ画像を入力し物体を認識しクラスを推定
                preds = self.model.predict(inputs, batch_size=1, verbose=1)

                # 推定結果をresultsへ抽出
                results = self.bbox_util.detection_out(preds)

                
                # resultsからクラスのラベル，信頼度，物体の最大・最小x,y座標を抽出
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                # 信頼度が0.6以上の推定結果のみを抽出
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                
                for i in range(top_conf.shape[0]):
                    x_min = int(round(top_xmin[i] * img.shape[1]))
                    y_min = int(round(top_ymin[i] * img.shape[0]))
                    x_max = int(round(top_xmax[i] * img.shape[1]))
                    y_max = int(round(top_ymax[i] * img.shape[0]))
                    score = top_conf[i]
                    label = int(top_label_indices[i])
                    label_name = self.voc_classes[label - 1]
                    display_txt = '{}, {:0.2f}'.format(label_name, score)
                    # バウンディングボックスを描画
                    frame = self.bounding_box(frame, x_min, y_min, x_max, y_max, label - 1)

                    text_top = (x_min, y_min - 10)
                    text_bot = (x_min + 80, y_min + 10)
                    text_pos = (x_min + 5, y_min + 5)
                    frame = cv2.putText(frame, display_txt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

                    if self.is_upload:  # [10-2-4].この行は絶対に書き換えない
                        # [10-2-5].検出されたラベルに応じて，Slackにアップロードするライブラリを呼び出す
                        if False:  # [10-2-6].この行を自分で書き換える
                            upload_photo_to_slack()  # [10-2-7].この関数に渡す引数を自分で書き換える
                            self.is_upload = False

            cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.imshow("Result", frame)
            #self.writer.write(image)##movie save

            # FPSの算出
            elapsedTime = time.time() - t1
            self.fps = "{:.0f} FPS".format(1/elapsedTime)


        cv2.destroyAllWindows()
    
main = detection()
main.main()
