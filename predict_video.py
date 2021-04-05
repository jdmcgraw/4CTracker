import numpy as np
from threading import Timer
import cv2
import pickle

import torch
import torch.nn as nn
import numpy as np
import time

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


class ModelViewer:

    def __init__(self, model_name, frame_source):
        self.model = DepthNet()
        self.project_model = model_name
        self.project_frames = frame_source
        self.model.load_state_dict(torch.load(f'{self.project_model}.net'))
        self.model.eval()

        with open(f'{self.project_frames}.rgbd', 'rb') as frames_in:
            self.frames = np.array(pickle.load(frames_in))

        self.current_frame_index = 0
        cv2.namedWindow('Tool')
        self.play()

        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                return

    def play(self):
        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = 0
        self.deliver_preview_frame(preview_size=512)
        self.current_frame_index += 1
        t = Timer(0.03, self.play)
        t.start()

    def deliver_preview_frame(self, preview_size, verbose=True):
        torch_frames = torch.from_numpy(self.frames).type(torch.float).reshape(-1, 4, TARGET_MODEL_SIZE,
                                                                                      TARGET_MODEL_SIZE)
        current_torch_frame = torch_frames[self.current_frame_index].reshape(1, 4, TARGET_MODEL_SIZE,
                                                                                   TARGET_MODEL_SIZE)
        start_time = time.time()
        x, y = self.model(current_torch_frame).data[0]
        #if verbose:
        #    print(f"Executing Model at {round(60/((time.time() - start_time) * 1000), 1)}Hz")
        im = self.frames[self.current_frame_index, :, :, :-1]
        im_resize = cv2.resize(im, (preview_size, preview_size))
        x_resize, y_resize = int(x * preview_size), int(y * preview_size)
        cv2.circle(im_resize, (x_resize, y_resize), 3, (0, 0, 255), 2)
        cv2.imshow('Tool', im_resize)


preview = ModelViewer("test2", "test2")
