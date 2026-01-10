import os
import time
import traceback
from typing import Optional, Tuple
import json

import cv2
import numpy as np

# ---- NEW: Aravis streamer import ----
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_stream import AravisCameraStreamer


class FrameData:
    def __init__(self, front_rgb, back_rgb, capture_time, receive_time):
        self.front_rgb = front_rgb
        self.back_rgb = back_rgb
        self.capture_time = capture_time
        self.receive_time = receive_time


# ---- NEW: tiny wrapper to mimic Insta360SharedMem.receive_image() ----
class AravisSingleCamera:
    """
    Drop-in-ish wrapper that exposes receive_image() like Insta360SharedMem,
    but returns a single camera image from AravisCameraStreamer using `camera_key`.
    """
    def __init__(self, camera_key: str):
        self.camera_key = camera_key
        self.streamer = AravisCameraStreamer()

        available = list(getattr(self.streamer, "_camera_map", {}).keys())
        if camera_key not in getattr(self.streamer, "_camera_map", {}):
            raise SystemExit(f"Camera key '{camera_key}' not found. Available: {available}")

        self.started = False

    def start(self):
        if not self.started:
            self.streamer.start()
            self.streamer.wait_until_ready(min_frames=1, timeout=5.0)
            self.started = True

    def stop(self):
        if self.started:
            self.streamer.stop()
            self.started = False

    def receive_image(self) -> np.ndarray:
        # returns (H,W,3) image
        frames, _ts = self.streamer.get_observation_window(1, include_timestamps=True)
        img = frames[self.camera_key][0]
        return img


class Insta360Calibrated:
    def __init__(
        self,
        camera,
        camera_resolution: Tuple[int, int] = (1920, 1920),
        latency: Optional[float] = 0.101,
        image_save_path: str = "./images",
        camera_calibration_save_path: str = "./camera_calibration",
    ):
        self.camera_resolution = camera_resolution
        self.latency = latency
        self.is_running = False

        # Initialize camera
        self.cap = camera

        # Calibration parameters
        self.DIM = None
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        self.maps = None
        self.calibrated = False
        self.objpoints = []
        self.imgpoints = []

        # path to save
        self.image_save_path = image_save_path
        self.camera_calibration_save_path = camera_calibration_save_path
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.camera_calibration_save_path, exist_ok=True)

    def get_camera_frame(self) -> Optional[FrameData]:
        """Capture a frame from the camera."""
        if not self.is_running:
            return None

        try:
            # ---- NEW: ensure Aravis is started if wrapper supports it ----
            if hasattr(self.cap, "start"):
                self.cap.start()

            ret = True
            frame = self.cap.receive_image()
            receive_time = time.monotonic()

            if not ret or frame is None:
                return None

            # ---- unchanged: keep front/back split API ----
            # For single Aravis camera, we'll just return the same image in both.
            if frame.ndim == 3 and frame.shape[2] == 3:
                front_frame = frame
                back_frame = frame
            else:
                # handle grayscale etc.
                front_frame = frame
                back_frame = frame

            if self.camera_resolution is not None:
                front_frame = cv2.resize(front_frame, self.camera_resolution)
                back_frame = cv2.resize(back_frame, self.camera_resolution)

            capture_time = receive_time - (self.latency or 0)

            return FrameData(
                front_rgb=front_frame,
                back_rgb=back_frame,
                capture_time=capture_time,
                receive_time=receive_time,
            )

        except Exception as e:
            print(f"Error capturing frame: {e}")
            traceback.print_exc()
            return None

    def calibrate_camera(self, chessboard_size=(9, 6), square_size=1.0, num_images=30):
        """
        Calibrate the fisheye camera and save the calibration parameters.
        """
        print(
            f"Starting calibration. Please show the {chessboard_size[0]}x{chessboard_size[1]} chessboard pattern..."
        )

        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_FIX_SKEW
            + cv2.fisheye.CALIB_FIX_K4
        )

        # Prepare object points
        objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        self.start_streaming()
        captured_images = 0
        last_capture_time = time.time() - 2

        while captured_images < num_images:
            frame_data = self.get_camera_frame()
            if frame_data is None:
                continue

            frame = frame_data.front_rgb
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray,
                chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display_frame = frame.copy()

            if ret and (time.time() - last_capture_time) > 1.0:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (3, 3),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                )

                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret)
                captured_images += 1
                last_capture_time = time.time()
                print(f"Captured image {captured_images}/{num_images}")

            cv2.putText(
                display_frame,
                f"Captured: {captured_images}/{num_images}. Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Calibration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyWindow("Calibration")

        if captured_images > 0:
            print("Computing calibration...")
            try:
                self.DIM = gray.shape[::-1]  # (width, height)

                N_OK = len(self.objpoints)
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
                self.K = np.eye(3, dtype=np.float64)
                self.D = np.zeros((4, 1), dtype=np.float64)

                rms, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(
                    self.objpoints,
                    self.imgpoints,
                    self.DIM,
                    self.K,
                    self.D,
                    None,
                    None,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                )

                calibration_data = {"DIM": self.DIM, "K": self.K.tolist(), "D": self.D.tolist()}

                with open(os.path.join(self.camera_calibration_save_path, "fisheye_calibration.json"), "w") as f:
                    json.dump(calibration_data, f, indent=2)

                print(f"\nCalibration successful! RMS error: {rms:.2f}")
                print("\nCalibration parameters:")
                print(f"DIM={self.DIM}")
                print(f"K=np.array({self.K.tolist()})")
                print(f"D=np.array({self.D.tolist()})")
                print("\nParameters saved to 'fisheye_calibration.json'")

                self.maps = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2
                )
                self.calibrated = True

            except Exception as e:
                print(f"Calibration failed: {e}")
                traceback.print_exc()
        else:
            print("Not enough images captured for calibration.")

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            return frame

        if self.maps is None:
            self.maps = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2
            )

        return cv2.remap(
            frame,
            self.maps[0],
            self.maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def load_calibration(self, calibration_file="fisheye_calibration.json"):
        try:
            with open(calibration_file, "r") as f:
                data = json.load(f)

            self.DIM = tuple(data["DIM"])
            self.K = np.array(data["K"])
            self.D = np.array(data["D"])
            self.calibrated = True
            self.maps = None

            print("Loaded calibration parameters:")
            print(f"DIM={self.DIM}")
            print(f"K=np.array({self.K.tolist()})")
            print(f"D=np.array({self.D.tolist()})")

        except Exception as e:
            print(f"Error loading calibration file: {e}")
            self.calibrated = False
            raise

    def start_streaming(self):
        self.is_running = True
        print("started")

    def stop_streaming(self):
        self.is_running = False
        # ---- NEW: stop Aravis wrapper if present ----
        if hasattr(self.cap, "stop"):
            self.cap.stop()
        print("stopped")


if __name__ == "__main__":
    # ---- Choose ONE camera key here and calibrate it ----
    # examples:
    #   camera_head_main_rgb
    #   camera_head_main_right_rgb
    camera_key = "camera_head_main_rgb"

    camera = AravisSingleCamera(camera_key=camera_key)

    cam = Insta360Calibrated(
        camera=camera,
        camera_resolution=None,  # keep native stream size (e.g., 480x300); set (W,H) if you want resize
        latency=0.0,
        image_save_path=f"./images_{camera_key}",
        camera_calibration_save_path=f"./camera_calibration_{camera_key}",
    )

    # IMPORTANT: chessboard_size is INNER corners (cols, rows)
    cam.calibrate_camera(chessboard_size=(8, 5), square_size=0.025, num_images=40)

    cam.stop_streaming()

# if __name__ == "__main__":
#     # Pick which camera to view
#     camera_key = "camera_head_main_right_rgb"
#     calibration_file = f"./camera_calibration_{camera_key}/fisheye_calibration.json"

#     camera = AravisSingleCamera(camera_key=camera_key)

#     cam = Insta360Calibrated(
#         camera=camera,
#         camera_resolution=None,  # keep native stream size
#         latency=0.0,
#         image_save_path=f"./images_{camera_key}",
#         camera_calibration_save_path=f"./camera_calibration_{camera_key}",
#     )

#     # 1) Load calibration
#     cam.load_calibration(calibration_file)

#     # 2) Start streaming loop
#     cam.start_streaming()
#     cv2.namedWindow("raw_vs_undistorted", cv2.WINDOW_NORMAL)

#     try:
#         while True:
#             frame_data = cam.get_camera_frame()
#             if frame_data is None:
#                 continue

#             raw = frame_data.front_rgb
#             und = cam.undistort_frame(raw)

#             # Make a side-by-side view (match heights if needed)
#             if raw.shape[:2] != und.shape[:2]:
#                 und = cv2.resize(und, (raw.shape[1], raw.shape[0]))

#             vis = np.concatenate([raw, und], axis=1)
#             cv2.putText(vis, "RAW", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.putText(vis, "UNDISTORTED", (raw.shape[1] + 10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             cv2.imshow("raw_vs_undistorted", vis)

#             key = cv2.waitKey(1) & 0xFF
#             if key in (27, ord("q")):
#                 break
#             elif key == ord("s"):
#                 # save a snapshot pair
#                 ts = int(time.time() * 1000)
#                 out_raw = os.path.join(cam.image_save_path, f"raw_{ts}.png")
#                 out_und = os.path.join(cam.image_save_path, f"und_{ts}.png")
#                 cv2.imwrite(out_raw, raw)
#                 cv2.imwrite(out_und, und)
#                 print(f"Saved: {out_raw}")
#                 print(f"Saved: {out_und}")

#     finally:
#         cam.stop_streaming()
#         cv2.destroyAllWindows()

