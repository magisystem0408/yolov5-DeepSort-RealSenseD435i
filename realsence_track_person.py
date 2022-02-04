import pyrealsense2 as rs
import cv2

import sys

import numpy as np

sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import math
import torch

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

WIDTH = 1280
FPS = 6
HEIGHT = 720

# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
attempt_download("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

device = select_device("cpu")
half = device.type != "cpu"  # half precision only supported on CUDA

model = attempt_load('yolov5/weights/yolov5s.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()


def realsence():
    try:
        # 3つの配列まで登録を可能にする
        while True:
            # フレーム待ち(Color & Depth)
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            # Depth画像
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            img0 = color_image.copy()

            img = letterbox(img0, 640, 32, True)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0

            frame_idx = 0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
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
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            if names[c] == "person":
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))

                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]

                                center_x = math.floor(bbox_left + (bbox_w / 2))
                                center_y = math.floor(bbox_top + (bbox_h / 2))

                                depth = depth_frame.get_distance(center_x, center_y)
                            
                                anotationList.append(
                                    [frame_idx, id, c, names[c], bbox_left, bbox_top, bbox_w, bbox_h, center_x, center_y,
                                    depth])

            result_image = annotator.result()

            if len(anotationList)>0:
                for anotation in anotationList:

                    print(anotation)

                    cv2.putText(result_image,str(anotation[10]),(int(anotation[8]),int(anotation[9])),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),3,cv2.LINE_AA)

            images = np.hstack((result_image, depth_color_image))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        # ストリーミング停止
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test = realsence()
    print(test)
