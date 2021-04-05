import numpy as np
from threading import Timer
import cv2
import pickle
import pyrealsense2 as rs
import os.path
from kmeans_clustering import KmeansClassifier


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class LabelingTool:

    def __init__(self, overwrite=False, frames_to_label=100, time_length=0, perform_sampling=True):
        self.project = "mouse"
        self.video_path = f"{self.project}.rgbd"
        self.label_path = f"{self.project}.labels"
        self.current_frame_index = 0
        self.current_key_index = 0
        self.current_frame = None
        self.current_display = None
        self.playback_speed = 0.1
        self.key_colors = [(0, 0, 255),
                           (0, 106, 255),
                           (0, 216, 255),
                           (0, 255, 182),
                           (144, 255, 0),
                           (255, 148, 0),
                           (255, 0, 72)]

        self.display_size = 512  # The size "we" view the image, regardless of actual image dimensions underneath

        self.frames = []
        self.model_frames = []
        self.key_points = ['head', 'body', 'tail']
        self.perform_sampling = perform_sampling

        self.playback = False
        self.color_view = True
        self.overwrite = overwrite
        cv2.namedWindow('Tool')
        cv2.setMouseCallback('Tool', self.on_mouse)

        if os.path.exists(self.video_path):
            self.read_frames()

        if self.perform_sampling:
            clustered_frames = KmeansClassifier(self.frames, clusters=frames_to_label)

            new_frames = []
            # This allows us to optionally sample around the clustered points, rather than just individually
            if time_length > 0:
                for i in clustered_frames.get_clusters():
                    for j in range(-time_length, time_length):
                        time_frame = i+j
                        if 0 <= time_frame < len(self.frames):
                            new_frames.append(self.frames[time_frame])
                self.frames = np.array(new_frames)
            else:
                self.frames = np.array([self.frames[k] for k in clustered_frames.get_clusters()])

        # Negative Ones Array
        self.frame_labels = np.ones(shape=(len(self.frames), len(self.key_points), 2)) * -1
        if os.path.exists(self.label_path):
            self.load_labels()

        self.current_frame_index = 0
        self.current_frame = self.frames[self.current_frame_index]
        self.deliver_preview_frame(self.current_frame_index)

        while True:
            key = cv2.waitKey(0)
            if key == 8:  # Backspace
                self.frame_labels[self.current_frame_index][self.current_key_index] = np.array([-1, -1])
                self.deliver_preview_frame(self.current_frame_index)
                self.deliver_preview_frame(self.current_frame_index)
            if key == ord(' '):
                self.playback = not self.playback
                t = Timer(self.playback_speed, self.play)
                t.start()
            if key == ord('1'):
                print(f"[MODE] View Toggled to {'Color' if not self.color_view else 'Depth'}")
                self.color_view = not self.color_view
                self.deliver_preview_frame(self.current_frame_index)
            if key == ord('.'):  # >
                self.current_key_index += 1
                self.current_key_index = clamp(self.current_key_index, 0, len(self.key_points)-1)
                self.deliver_preview_frame(self.current_frame_index)
            if key == ord(','):  # <
                self.current_key_index -= 1
                self.current_key_index = clamp(self.current_key_index, 0, len(self.key_points)-1)
                self.deliver_preview_frame(self.current_frame_index)
            if key == ord('q'):
                self.playback = False
                print("[QUIT] Closing")
                return

    def play(self):
        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = 0
        self.deliver_preview_frame(self.current_frame_index)
        self.current_frame_index += 1
        if self.playback:
            t = Timer(self.playback_speed, self.play)
            t.start()

    def get_color_frame(self, frame):
        return self.frames[frame, :, :, :-1].copy()

    def get_depth_frame(self, frame):
        return self.frames[frame, :, :, -1].copy()

    def save_labels(self):
        with open(self.label_path, 'wb') as labels:
            pickle.dump(self.frame_labels, labels)

    def load_labels(self):
        if self.overwrite:
            return
        with open(self.label_path, 'rb') as labels:
            self.frame_labels = pickle.load(labels)
        if len(self.frame_labels) > len(self.frames):
            self.frames = []

    def read_frames(self):
        print("[INFO] Reading Frames...")
        with open(self.video_path, 'rb') as in_file:
            self.frames = np.array(pickle.load(in_file), dtype=np.uint8)
        print(f"[INFO] RGBD Video: {len(self.frames)} Frames")

    def deliver_preview_frame(self, frame=0):
        if self.color_view:
            self.current_display = cv2.resize(self.get_color_frame(frame), (self.display_size, self.display_size))
        else:
            self.current_display = cv2.resize(self.get_depth_frame(frame), (self.display_size, self.display_size))
            self.current_display = 1 - cv2.cvtColor(self.current_display, cv2.COLOR_GRAY2RGB)

        text_pos = (int(0.03 * self.display_size), int(0.05 * self.display_size))
        color = (255, 255, 255) if self.current_frame_index < len(self.frames)-1 else (0, 0, 255)
        if self.playback:
            color = (0, 255, 0)
        cv2.putText(self.current_display, f'[Frame {self.current_frame_index+1}/{len(self.frames)}]',
                    text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        text_pos = (int(0.03 * self.display_size), int(0.11 * self.display_size))
        text_color = (0, 144, 255) if self.color_view else (255, 0, 255)
        cv2.putText(self.current_display, f"[{'RGB' if self.color_view else 'Depth'} View]",
                    text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

        if self.current_key_index > 0:
            text_pos = (int(0.1 * self.display_size), int(0.95 * self.display_size))
            text_color = self.key_colors[self.current_key_index - 1]
            cv2.putText(self.current_display, f"{self.key_points[self.current_key_index - 1]}".ljust(10),
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

        text_pos = (int(0.4 * self.display_size), int(0.95 * self.display_size))
        text_color = self.key_colors[self.current_key_index]
        cv2.putText(self.current_display, f"<{self.key_points[self.current_key_index]}>".ljust(10),
                    text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

        if self.current_key_index < len(self.key_points)-1:
            text_pos = (int(0.7 * self.display_size), int(0.95 * self.display_size))
            text_color = self.key_colors[self.current_key_index + 1]
            cv2.putText(self.current_display, f"{self.key_points[self.current_key_index + 1]}".ljust(10),
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

        # Draw the current frame keypoints
        for i, keypoint in enumerate(self.frame_labels[self.current_frame_index]):
            pos = tuple(int(p * self.display_size) for p in keypoint)
            key_color = self.key_colors[i]
            cv2.circle(self.current_display, pos, 4, key_color, 1)

            cv2.line(self.current_display, (pos[0] + 3, pos[1]), (pos[0] + 8, pos[1]), key_color)
            cv2.line(self.current_display, (pos[0] - 3, pos[1]), (pos[0] - 8, pos[1]), key_color)
            cv2.line(self.current_display, (pos[0], pos[1] + 3), (pos[0], pos[1] + 8), key_color)
            cv2.line(self.current_display, (pos[0], pos[1] - 3), (pos[0], pos[1] - 8), key_color)

        cv2.imshow('Tool', self.current_display)

    def on_mouse(self, event, x, y, flags, param):
        if not (x and y) or self.playback:
            return

        self.current_display = self.current_frame.copy()
        if event == 1:  # Click
            self.current_frame = self.frames[self.current_frame_index]
            self.frame_labels[self.current_frame_index][self.current_key_index] = (x/self.display_size,
                                                                                   y/self.display_size)
            self.current_frame_index = self.current_frame_index + 1
            self.current_frame_index = int(clamp(self.current_frame_index, 0, len(self.frames) - 1))
            self.deliver_preview_frame(self.current_frame_index)
            self.save_labels()

            if self.current_frame_index == len(self.frame_labels)-1 and self.current_key_index < len(self.key_points)-1:
                self.current_frame_index = 0
                self.current_key_index += 1
                self.current_key_index = clamp(self.current_key_index, 0, len(self.key_points) - 1)
                self.deliver_preview_frame(self.current_frame_index)

        if abs(flags) > 1:  # Scroll
            self.current_frame = self.frames[self.current_frame_index]
            self.current_frame_index = self.current_frame_index + (np.sign(flags))
            self.current_frame_index = int(clamp(self.current_frame_index, 0, len(self.frames) - 1))
            self.deliver_preview_frame(self.current_frame_index)


tool = LabelingTool(overwrite=True, perform_sampling=True)
