import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

PREVIEW_FRAME_SIZE = (512, 512)  # Visualization Frame Size
TARGET_MODEL_SIZE = 128


class VideoRecorder:

    def __init__(self, model_name):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.model_name = model_name

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.final_color_frames = []
        self.final_depth_frames = []
        print(f'[INFO] Device: {self.device}')

    def record_rgbd(self, preview=True, write=True):

        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        # Aligning depth to color image
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Colorize the Depth Frame
        colorizer_jet = rs.colorizer()

        print("[INFO] Recording Started...")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame_raw = aligned_frames.get_depth_frame()
                color_frame_raw = aligned_frames.get_color_frame()

                color_frame = np.asanyarray(color_frame_raw.get_data())
                depth_frame_col = np.asanyarray(colorizer_jet.colorize(depth_frame_raw).get_data())
                depth_frame_raw = np.asanyarray(depth_frame_raw.get_data())

                color_frame = cv2.resize(color_frame, (depth_frame_col.shape[1], depth_frame_col.shape[0]))

                # Stack Frames for Display
                camera_images = np.hstack((color_frame, depth_frame_col))

                # Display Images
                if preview:
                    cv2.namedWindow(f'Project: {self.model_name} Preview', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'Project: {self.model_name} Preview', camera_images)

                # Save Video Frames
                if write:
                    # Resize frames to model input size
                    color_frame_resized = cv2.resize(color_frame, (TARGET_MODEL_SIZE, TARGET_MODEL_SIZE))
                    depth_frame_raw_resized = cv2.resize(depth_frame_raw, (TARGET_MODEL_SIZE, TARGET_MODEL_SIZE))

                    # Save Frames
                    self.final_color_frames.append(color_frame_resized)
                    self.final_depth_frames.append(depth_frame_raw_resized)

                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    print("[INFO] Recording Stopped")
                    break

        except Exception as e:
            raise e
        finally:
            self. pipeline.stop()
            if write:
                self.save_rgbd_frames()

    def save_rgbd_frames(self):
        num_frames = len(self.final_color_frames)
        self.final_depth_frames = np.array(self.final_depth_frames, np.uint8)
        self.final_color_frames = np.array(self.final_color_frames, np.uint8)
        self.final_depth_frames = np.expand_dims(self.final_depth_frames, axis=-1)

        print(f'[INFO] Saving {num_frames} frames...')
        final_frames = np.append(self.final_color_frames, self.final_depth_frames, axis=-1)
        final_frames = np.reshape(final_frames, (num_frames, TARGET_MODEL_SIZE, TARGET_MODEL_SIZE, 4))

        with open(f'{self.model_name}.rgbd', 'wb') as out_file:
            pickle.dump(final_frames, out_file)

        print(f"[DONE] Recording Saved to {self.model_name}.rgbd")

        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                return


if __name__ == "__main__":
    vr = VideoRecorder('test2')
    vr.record_rgbd(preview=True, write=True)

