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
from yeelight import Bulb


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

        self.bulb = Bulb("192.168.1.3")
        self.bulb.set_rgb(255, 0, 0, )

        self.authentication = False
        self.main()

    def main(self):

        def _getLength(x1, y1, x2, y2):
            return math.hypot(x2 - x1, y2 - y1)

        def _calcAngle(right_x, right_y, middle_x, middle_y, top_x, top_y):
            radians = np.arctan2(top_y - middle_y, top_x - middle_x) - np.arctan2(right_y - middle_y,
                                                                                  right_x - middle_x)
            angle = np.abs(radians * 180 / np.pi)
            if angle > 180:
                angle = 360 - angle
            return angle

        def findHand(img, draw=True):
            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(img)
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms,
                                                   self.mpHands.HAND_CONNECTIONS)
            return img

        def findPosition(img, handNo=0, draw=True, width=None, height=None):
            lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return lmList

        try:
            # 3つの配列まで登録を可能にする
            # cap = cv2.VideoCapture(1)
            registor_person = []
            auth_counter = True
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

                if len(anotationList) != 0:
                    for anotation in anotationList:
                        if len(registor_person) <= 0 and anotation[3] == "person" and (
                                int(anotation[10]) > 0 and int(anotation[10] <= 100)):
                            registor_person.append(anotation)
                            self.authentication = True
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

                if not self.authentication:
                    cv2.putText(result_image, "AUTH MODE", (30, 100),
                                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3, cv2.LINE_AA)

                    cv2.imshow("authentication", result_image)

                if len(mp_pred_rect) != 0:

                    # もしウィンドウが開いたらauthの画面は消す
                    if auth_counter:
                        cv2.destroyWindow("authentication")
                        auth_counter =False

                    x, y = mp_pred_rect[0][0], mp_pred_rect[0][1]
                    width, height = mp_pred_rect[0][2], mp_pred_rect[0][3]

                    target_person = imgRGB[y:y + height, x:x + width]

                    mp_pred = findHand(target_person)
                    target_image = cv2.cvtColor(mp_pred, cv2.COLOR_RGB2BGRA)

                    lmList = findPosition(img=mp_pred, width=width, height=height)

                    if len(lmList) != 0:
                        # 親指
                        right_thumb_tip_x, right_thumb_tip_y = lmList[4][1], lmList[4][2]
                        # 親指のした
                        right_thumb_ip_x, right_thum_ip_y = lmList[3][1], lmList[3][2]
                        # 更にした
                        right_thumb_cmc_x, right_thum_cmc_y = lmList[2][1], lmList[2][1]

                        # 人差し指
                        right_index_finger_tip_x, right_index_finger_tip_y = lmList[8][1], lmList[8][2]
                        # 人差し指のした
                        right_index_finger_dip_x, right_index_finger_dip_y = lmList[7][1], lmList[7][2]
                        # 人差しの付け根より一つ上
                        right_index_finger_pip_x, right_index_finger_pip_y = lmList[6][1], lmList[6][2]
                        # 人差しの付け根より上
                        right_index_finger_mcp_x, right_index_finger_mcp_y = lmList[5][1], lmList[5][2]

                        # 中指
                        right_middle_finger_tip_x, right_middle_finger_tip_y = lmList[12][1], lmList[12][2]
                        # 中指のした
                        right_middle_finger_dip_x, right_middle_finger_dip_y = lmList[11][1], lmList[11][2]
                        # 中指の付け根より一つ上
                        right_middle_finger_pip_x, right_middle_finger_pip_y = lmList[10][1], lmList[10][2]
                        # 中指の付け根
                        right_middle_finger_mcp_x, right_middle_finger_mcp_y = lmList[9][1], lmList[9][2]

                        # 薬指
                        right_ring_finger_tip_x, right_ring_finger_tip_y = lmList[16][1], lmList[16][2]
                        # 薬指のした
                        right_ring_finger_dip_x, right_ring_finger_dip_y = lmList[15][1], lmList[15][2]
                        # 薬指の付け根のしたより一つ上
                        right_ring_finger_pip_x, right_ring_finger_pip_y = lmList[14][1], lmList[14][2]
                        # 薬指の付け根
                        right_ring_finger_mcp_x, right_ring_finger_mcp_y = lmList[13][1], lmList[13][2]

                        # 小指
                        right_pinky_tip_x, right_pinky_tip_y = lmList[20][1], lmList[20][2]
                        # 小指のした
                        right_pinky_dip_x, right_pinky_dip_y = lmList[19][1], lmList[19][2]
                        # 小指の付け根より一つ上
                        right_pinky_pip_x, right_pinky_pip_y = lmList[18][1], lmList[18][2]
                        # 小指の付け根
                        right_pinky_mcp_x, right_pinky_mcp_y = lmList[17][1], lmList[17][2]

                        # 長さ判定
                        thumb_index_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                        right_index_finger_tip_x, right_index_finger_tip_y)
                        thumb_middle_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                         right_middle_finger_tip_x, right_middle_finger_tip_y)
                        thumb_ring_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                       right_ring_finger_tip_x, right_ring_finger_tip_y)

                        thumb_pinky_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                        right_pinky_tip_x, right_pinky_tip_y, )

                        # 角度計算(大回りよう)
                        thumb_angle = _calcAngle(right_thumb_cmc_x, right_thum_cmc_y,
                                                 right_thumb_ip_x, right_thum_ip_y,
                                                 right_thumb_tip_x, right_thumb_tip_y
                                                 )

                        index_angle = _calcAngle(right_index_finger_mcp_x, right_index_finger_mcp_y,
                                                 right_index_finger_pip_x, right_index_finger_pip_y,
                                                 right_index_finger_tip_x, right_index_finger_tip_y
                                                 )

                        middle_angle = _calcAngle(right_middle_finger_mcp_x, right_middle_finger_mcp_y,
                                                  right_middle_finger_pip_x, right_middle_finger_pip_y,
                                                  right_middle_finger_tip_x, right_middle_finger_tip_y
                                                  )
                        ring_angle = _calcAngle(right_ring_finger_mcp_x, right_ring_finger_mcp_y,
                                                right_ring_finger_pip_x, right_ring_finger_pip_y,
                                                right_ring_finger_tip_x, right_ring_finger_tip_y,
                                                )

                        pinky_angle = _calcAngle(right_pinky_mcp_x, right_pinky_mcp_y,
                                                 right_pinky_pip_x, right_pinky_pip_y,
                                                 right_pinky_tip_x, right_pinky_tip_y
                                                 )

                        # 角度計算(小回り用)
                        index_angle_2 = _calcAngle(right_index_finger_pip_x, right_index_finger_pip_y,
                                                   right_index_finger_dip_x, right_middle_finger_dip_y,
                                                   right_index_finger_tip_x, right_index_finger_tip_y,
                                                   )

                        middle_angle_2 = _calcAngle(right_middle_finger_pip_x, right_middle_finger_pip_y,
                                                    right_middle_finger_dip_x, right_middle_finger_dip_y,
                                                    right_middle_finger_tip_x, right_middle_finger_tip_y
                                                    )
                        ring_angle_2 = _calcAngle(right_ring_finger_pip_x, right_ring_finger_pip_y,
                                                  right_ring_finger_dip_x, right_ring_finger_dip_y,
                                                  right_ring_finger_tip_x, right_ring_finger_tip_y
                                                  )
                        pinky_angle_2 = _calcAngle(right_pinky_pip_x, right_pinky_tip_y,
                                                   right_pinky_dip_x, right_pinky_dip_y,
                                                   right_pinky_tip_x, right_pinky_tip_y
                                                   )
                    cv2.putText(target_image, "TARGET", (20, 70),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imshow('registor', target_image)


                    cv2.imshow("fuction", result_image)

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
