import numpy as np
from threading import Timer
import cv2
import pickle

import torch
import torch.nn as nn
import numpy as np
import time
import pyrealsense2 as rs

TARGET_MODEL_SIZE = 128


class VideoWriter:
    def __init__(self, path, frame_size, codec="mp4v", fps=60.0, color=True):
        codec = cv2.VideoWriter_fourcc(*codec)
        self.stream = cv2.VideoWriter(path, codec, fps, frame_size, color)

    def write(self, frame):
        self.stream.write(frame)

    def close(self):
        self.stream.release()
        return not self.stream.isOpened()


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear((30**2) * 32, 512, bias=True),
            nn.LeakyReLU())
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU())
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = self.drop_out(out)
        out = self.fc3(out)
        return out


class LiveModelViewer:

    def __init__(self, model_name):
        self.model = DepthNet()
        self.preview_size = (720, 720)
        self.project_model = "test2"
        self.model.load_state_dict(torch.load(f'{self.project_model}.net'))
        self.model.eval()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.model_name = model_name

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        print(f'[INFO] Device: {self.device}')

        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        print("[INFO] Stream Started...")

        # Aligning depth to color image
        self.align = rs.align(rs.stream.color)

        cv2.namedWindow(f'Project: {self.model_name} Preview')

        while True:
            self.deliver_preview_frame(self.preview_size)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                print("[INFO] Stream Stopped")
                return

    def deliver_preview_frame(self, preview_size, verbose=True):
        current_frame = self.get_current_frame()
        current_torch_frame = torch.from_numpy(current_frame).type(torch.float).reshape(1, 4, TARGET_MODEL_SIZE,
                                                                                        TARGET_MODEL_SIZE)
        start_time = time.time()
        x, y = self.model(current_torch_frame).data[0]

        im = current_frame[0, :, :, :-1]
        fps = round(60 / ((time.time() - start_time) * 1000))
        im_resize = cv2.resize(im, preview_size)
        cv2.putText(im_resize, f'{fps} fps'.rjust(8), (-20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        x_resize, y_resize = int(x * preview_size[0]), int(y * preview_size[1])
        cv2.circle(im_resize, (x_resize, y_resize), 3, (0, 0, 255), 2)
        cv2.imshow(f'Project: {self.model_name} Preview', im_resize)

    def get_current_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame_raw = aligned_frames.get_depth_frame()
        color_frame_raw = aligned_frames.get_color_frame()

        color_frame = np.asanyarray(color_frame_raw.get_data())
        depth_frame_raw = np.asanyarray(depth_frame_raw.get_data())
        color_frame = cv2.resize(color_frame, (depth_frame_raw.shape[1], depth_frame_raw.shape[0]))

        final_color_frame = np.array(cv2.resize(color_frame, (TARGET_MODEL_SIZE, TARGET_MODEL_SIZE)), np.uint8)
        final_depth_frame = np.array(cv2.resize(depth_frame_raw, (TARGET_MODEL_SIZE, TARGET_MODEL_SIZE)), np.uint8)
        final_depth_frame = np.expand_dims(final_depth_frame, axis=-1)

        joined_frame = np.append(final_color_frame, final_depth_frame, axis=-1)
        joined_frame = np.expand_dims(joined_frame, axis=0)
        return joined_frame


preview = LiveModelViewer("test2")
