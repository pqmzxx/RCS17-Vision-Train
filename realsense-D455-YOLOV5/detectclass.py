import os
import shutil
import time
import cv2
import torch
import numpy as np
from numpy import random
import pyrealsense2 as rs
from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    xyxy2xywh, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox


class Detect:
    def __init__(self, weights, device, iou, source, view_img, save_txt, img_size):
        self.device = select_device(device)
        self.weights = weights
        self.iou = iou
        self.source = source
        self.view_img = view_img
        self.save_txt = save_txt
        self.img_size = img_size
        self.model = None
        self.half = None
        self.pipeline = None
        self.align = None
        self.names = None
        self.colors = None

    def initialize(self, save_dir):
        set_logging()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)
        imgsz = check_img_size(self.img_size, s=self.model.stride.max())
        if self.device.type != 'cpu':
            self.model.half()  # to FP16
            self.half = True

        # Initialize RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, conf):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No color or depth frame available.")
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess image
        img = letterbox(color_image, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize

        img = torch.from_numpy(img).to(self.device).unsqueeze(0)
        if self.half:
            img = img.half()  # to FP16

        # Run inference
        pred = self.model(img, augment=conf['augment'])[0]
        pred = non_max_suppression(pred, conf['conf_thres'], self.iou, classes=conf['classes'], agnostic=conf['agnostic_nms'])
        return color_image, depth_frame, pred,img

    def plot_and_save(self, frame, depth_frame, pred, img, save_dir):
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"Unexpected frame shape: {frame.shape}")
            return

        for i, det in enumerate(pred):  # detections per image
            im0 = frame.copy()
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(im0.shape)[[1, 0, 1, 0]]).view(-1).tolist()  # normalized xywh
                    mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2), int((int(xyxy[1]) + int(xyxy[3])) / 2)]  # middle pixel
                    min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))  # search range

                    # Depth estimation
                    randnum = 40
                    distance_list = []
                    for _ in range(randnum):
                        bias = random.randint(-min_val // 4, min_val // 4)
                        dist = depth_frame.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))
                        if dist:
                            distance_list.append(dist)
                    distance_list = np.array(distance_list)
                    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]

                    label = f'{self.names[int(cls)]} {np.mean(distance_list):.2f}m'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

            # Stream results
            if self.view_img:
                cv2.imshow('Detection', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, 'result.jpg'), im0)

    def release(self):
        # Stop the pipeline and release resources
        self.pipeline.stop()
