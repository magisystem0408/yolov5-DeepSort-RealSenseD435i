import sys
import cv2
import numpy as np

sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import math
import torch

import pyrealsense2 as rs
import mediapipe as mp


class RealSense(object):
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.fps = 6
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

        self.cfg = get_config()
        self.cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        attempt_download("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
                         repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        self.device = select_device("cpu")
        self.half = self.device.type != "cpu"
        self.model = attempt_load('yolov5/weights/yolov5s.pt', map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.half:
            self.model.half()

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.main()

    def main(self):
        print("test")
        try:
            # 3つの配列まで登録を可能にする
            # cap = cv2.VideoCapture(1)
            registor_person = []
            while True:
                # フレーム待ち(Color & Depth)
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                imgRGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Depth画像
                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_color_image = np.asanyarray(depth_color_frame.get_data())

                ##################

                img0 = color_image.copy()

                img = letterbox(img0, 640, 32, True)[0]
                # Convert
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)

                img = img.half() if self.half else img.float()
                img /= 255.0

                frame_idx = 0

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # yolov5_deepsortで推論する
                pred = self.model(img, augment=False)[0]
                pred = non_max_suppression(
                    pred, 0.4, 0.5, agnostic=False)

                # 箱一つに対して処理をしている
                for i, det in enumerate(pred):  # detections per image
                    im0 = img0
                    # print string
                    annotator = Annotator(im0, line_width=2, pil=not ascii)

                    anotationList = []
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]
                        # pass detections to deepsort
                        outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                        if len(outputs) > 0:
                            for j, (output, conf) in enumerate(zip(outputs, confs)):
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]

                                c = int(cls)  # integer class
                                label = f'{id} {self.names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]

                                center_x = math.floor(bbox_left + (bbox_w / 2))
                                center_y = math.floor(bbox_top + (bbox_h / 2))

                                center_mask = np.array(
                                    [list(item[center_x - 5: center_x + 5]) for item in
                                     depth_color_image[center_y - 5: center_y + 5]])

                                depth = np.median(center_mask)

                                anotationList.append(
                                    [frame_idx, id, c, self.names[c], bbox_left, bbox_top, bbox_w, bbox_h, center_x,
                                     center_y,
                                     depth])

                result_image = annotator.result()

                # 推論場所
                mp_pred_rect = []

                if len(anotationList) > 0:
                    for anotation in anotationList:
                        if len(registor_person) <= 0 and anotation[3] == "person" and (
                                int(anotation[10]) > 0 and int(anotation[10] <= 100)):
                            registor_person.append(anotation)
                            print("登録しました")

                        # 登録されているとき
                        if len(registor_person) != 0 and (int(registor_person[0][2]) == int(anotation[2])):
                            # print("一致しています")
                            # print(anotation)
                            bbox_x, bbox_y = int(anotation[4]), int(anotation[5])
                            width, height = int(anotation[6]), int(anotation[7])
                            mp_pred_rect.append([bbox_x, bbox_y, width, height])
                        cv2.putText(result_image, str(anotation[10]), (int(anotation[8]), int(anotation[9])),
                                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3, cv2.LINE_AA)

                cv2.imshow("te", result_image)

                if len(mp_pred_rect) != 0:
                    x, y = mp_pred_rect[0][0], mp_pred_rect[0][1]
                    width, height = mp_pred_rect[0][2], mp_pred_rect[0][3]

                    target_person = imgRGB[y:y + height, x:x + width]
                    mp_pred = self.hands.process(target_person)

                    target_image = cv2.cvtColor(target_person, cv2.COLOR_RGB2BGRA)

                    if mp_pred.multi_hand_landmarks:
                        for handLms in mp_pred.multi_hand_landmarks:
                            for id, lm in enumerate(handLms.landmark):
                                h, w, c = result_image.shape
                                print(h, w, c)
                                # print(lm.x,lm.y)

                                cx, cy = int(lm.x * width), int(lm.y * height)
                                print(cx, cy)

                                cv2.circle(target_image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    cv2.imshow('target', target_image)

                    # mpDraw.draw_landmarks(imgRGB,handLms,mpHands.HAND_CONNECTIONS)

                # print(registor_person)
                # 表示
                # images =np.hstack((result_image,depth_color_image))
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                #

                # cv2.imshow('RealSense', result_image)

                if cv2.waitKey(1) & 0xff == 27:
                    break

        finally:
            # ストリーミング停止
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    RealSense()
