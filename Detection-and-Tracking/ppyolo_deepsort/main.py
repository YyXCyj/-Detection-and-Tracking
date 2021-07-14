# from ppyolo import get_object_position_ppyolo
from detector_new import *
from deep_sort.detection import *
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from collections import deque
from ResNet_ReID_paddle.ResNet50_ReID import Net
# from GhostNet_ReID_paddle.GhostNet_ReID import GhostNet
# from ShuffleNet_ReID_paddle.ShuffleNet_ReID import *

import paddle
import time
import os
import cv2
import numpy as np
from PIL import Image


def color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

k=1

class Detector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "../yolo_obj/final.pdparams"  # 模型文件的目录
        # try:
        self.net = Net(num_classes=751, reid=True)
        # paddle.set_device("gpu")
        static_dict = paddle.load("D:/ProgramCode/Detection-and-Tracking/yolo-obj/final.pdparams")
        self.net.set_state_dict(static_dict)
        self.net.eval()

        max_cosine_distance = 0.4  # 余弦距离的控制阈值
        nn_budget = 100
        self.nms_max_overlap = 0.5

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)  # 实例化追踪器对象
        # except:
        #     print("读取模型失败，请检查文件路径并确保无中文文件夹！")

    def run(self,frame):
        image = frame.copy()
        # h, w, c = image.shape

        B,G,R = cv2.split(image.copy())
        EB=cv2.equalizeHist(B)
        EG=cv2.equalizeHist(G)
        ER=cv2.equalizeHist(R)
        img__detect=cv2.merge((EG, ER, EB))
        results, confidences, features = get_object_position_new(image, img__detect, 0.0, self.net)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(results, confidences, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # 返回在非极大值抑制下幸存的检测指数
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        dets = [] #x,y,x+w,y+h,conf,0
        boxes = []  # 存放追踪到的标记框 x,y,w,h
        indexIDs = []  # 存放追踪到的序号
        cls_IDs = []  # 存放追踪到的类别 0,0,0

        for det in detections:
            bbox = det.to_tlbr()
            dets.append([int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]),det.confidence,0])
            # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            id = int(track.track_id)

            # color = color_for_labels(id)
            # t_size = cv2.getTextSize(str(track.track_id), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            # cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
            # cv2.rectangle(image, (int(x), int(y)), (int(x + t_size[0] + 3), int(y + t_size[1] + 4)), color, -1)
            # cv2.putText(image, str(id), (int(x), int(y +t_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)

            boxes.append([x,y,w,h])
            indexIDs.append(id)
            cls_IDs.append(0)

        # count = len(results)
        # cv2.putText(image, "Person be detected: " + str(count), (int(20), int(20)), 0, 5e-3 * 200, (0, 255, 0), 1)
        # i = str('%05d' % k)
        # cv2.imwrite("D:/ProgramCode/ppyolo_deepsort/results/{}.jpg".format(i), image)

        return dets, boxes, indexIDs, cls_IDs


if __name__ == '__main__':
    video_name="D:/ProgramCode/ppyolo_deepsort/video/SoftwareCup-02.avi"
    cap=cv2.VideoCapture(video_name)
    detector_model = Detector()
    if cap.isOpened():
        while True:
            flag, image = cap.read()   # 获取画面
            if not flag: break
            image = cv2.flip(image, 1)  # 左右翻转
            dets, boxes, indexIDs, cls_IDs = detector_model.run(image)
            print("第",k,"帧：\n",len(dets),dets,'\n',len(boxes),boxes,'\n',len(indexIDs),indexIDs,'\n',len(cls_IDs),cls_IDs)
            k+=1
