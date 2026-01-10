import os
import subprocess  # Import subprocess to run shell commands from Python
import time
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import json

from Insta360Camera.Insta360_x4_client import Insta360SharedMem

# from Insta360_x4_client import Insta360SharedMem


class FrameData:
    def __init__(self, front_rgb, back_rgb, capture_time, receive_time):
        self.front_rgb = front_rgb 
        self.back_rgb = back_rgb
        self.capture_time = capture_time 
        self.receive_time = receive_time 

class Insta360Calibrated:
    def __init__(
        self,
        camera,
        camera_resolution: Tuple[int, int] = (1920, 1920),  # Square resolution for fisheye
        latency: Optional[float] = 0.101,
        image_save_path: str = './images',
        camera_calibration_save_path: str = './camera_calibration'
    ):
        self.camera_resolution = camera_resolution
        self.latency = latency
        self.is_running = False
        
        # Initialize camera
        self.cap = camera 
            
        # Set camera properties
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        # self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings
        # actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print(f"Camera initialized with:")
        # print(f"Resolution: {int(actual_width)}x{int(actual_height)}")
        # print(f"FPS: {int(actual_fps)}")

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
        self.camera_calibration_save_path=camera_calibration_save_path
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.camera_calibration_save_path, exist_ok=True)

    def get_camera_frame(self) -> Optional[FrameData]:
        """Capture a frame from the camera."""
        if not self.is_running:
            return None
        
        try:
            # ret, frame = self.cap.read()

            ret = True 
            frame = self.cap.receive_image()

            receive_time = time.monotonic()
      
            if not ret or frame is None:
                return None
            
            front_frame= frame[:, :frame.shape[1] // 2]
            back_frame= frame[:, frame.shape[1] // 2:]
            if self.camera_resolution is not None:
                frame = cv2.resize(frame, self.camera_resolution)
                
            capture_time = receive_time - (self.latency or 0)
            
            return FrameData(
                front_rgb=front_frame,
                back_rgb=back_frame,
                capture_time=capture_time,
                receive_time=receive_time
            )
        
        except Exception as e:
            print(f"Error capturing frame: {e}")
            traceback.print_exc()
            return None

    def calibrate_camera(self, chessboard_size=(9, 6), square_size=1.0, num_images=30):
        """
        Calibrate the fisheye camera and save the calibration parameters.
        """
        print(f"Starting calibration. Please show the {chessboard_size[0]}x{chessboard_size[1]} chessboard pattern...")
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
      
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_K4+cv2.fisheye.CALIB_FIX_K3  # Optional: You may also fix K3
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_FIX_K4+cv2.fisheye.CALIB_FIX_K3  # Optional: You may also fix K3
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_FIX_K4 # Optional: You may also fix K3

        # Prepare object points
        objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        self.start_streaming()
        captured_images = 0
        last_capture_time = time.time() - 2
        
        while captured_images < num_images:
            frame_data = self.get_camera_frame()
            if frame_data is None:
                continue
                
            frame = frame_data.front_rgb
            # frame = frame_data.rgb
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(
                gray, 
                chessboard_size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            display_frame = frame.copy()
            
            if ret and (time.time() - last_capture_time) > 1.0:
                corners = cv2.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
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
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("Calibration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Calibration")
        
        if captured_images > 0:
            print("Computing calibration...")
            try:
                # Get image dimensions
                self.DIM = gray.shape[::-1]  # (width, height)
                
                # Calibrate camera
                N_OK = len(self.objpoints)
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                self.K = np.eye(3, dtype=np.float64)  # Identity matrix
                self.D = np.zeros((4, 1), dtype=np.float64)  # 4 distortion coefficients
                print(calibration_flags)
                rms, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(
                    self.objpoints,
                    self.imgpoints,
                    self.DIM,
                    self.K,
                    self.D,
                    None,
                    None,
                    # rvecs,
                    # tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                )
                # cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER

                # Save calibration parameters
                calibration_data = {
                    'DIM': self.DIM,
                    'K': self.K.tolist(),
                    'D': self.D.tolist()
                }
                
                with open(os.path.join(self.camera_calibration_save_path,'fisheye_calibration.json'), 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                print(f"\nCalibration successful! RMS error: {rms:.2f}")
                print("\nCalibration parameters:")
                print(f"DIM={self.DIM}")
                print(f"K=np.array({self.K.tolist()})")
                print(f"D=np.array({self.D.tolist()})")
                print("\nParameters saved to 'fisheye_calibration.json'")
                
                # Initialize undistortion maps
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
        """Undistort a frame using the saved calibration parameters."""
        if not self.calibrated:
            return frame
        
        if self.maps is None:
            # Initialize maps if not already done
            self.maps = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2
            )
        
        return cv2.remap(frame, self.maps[0], self.maps[1], 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT)

    def load_calibration(self, calibration_file='fisheye_calibration.json'):
        """Load calibration parameters from a file."""
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                
            self.DIM = tuple(data['DIM'])
            self.K = np.array(data['K'])
            self.D = np.array(data['D'])
            self.calibrated = True
            self.maps = None  # Will be initialized on first undistortion
            
            print("Loaded calibration parameters:")
            print(f"DIM={self.DIM}")
            print(f"K=np.array({self.K.tolist()})")
            print(f"D=np.array({self.D.tolist()})")
            
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            exit()
            self.calibrated = False

    def convert_fisheye_to_pinhole(self, balance=0.0, dim2=None, dim3=None):
        """
        Convert fisheye camera parameters to approximate pinhole camera parameters.
        
        Args:
            balance (float): Balance between original (0.0) and zero distortion (1.0)
            dim2 (tuple): Optional new image dimensions (width, height)
            dim3 (tuple): Optional target dimensions for the undistorted image
        
        Returns:
            dict: Containing new camera matrix (K_pinhole) and distortion coefficients (D_pinhole)
        """
        if not self.calibrated:
            raise RuntimeError("Camera must be calibrated first!")
            
        dim1 = self.DIM
        if dim2 is None:
            dim2 = dim1
        if dim3 is None:
            dim3 = dim1

        # Calculate optimal new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, 
            dim2, 
            np.eye(3), 
            balance=balance,
            new_size=dim3,
            fov_scale=1.0
        )
        
        # Calculate undistortion and rectification transformation map
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        #     self.K, self.D, 
        #     np.eye(3), 
        #     new_K, 
        #     dim3, 
        #     cv2.CV_16SC2
        # )
        
        # Store the pinhole approximation parameters
        self.K_pinhole = new_K
        # For pinhole model, initialize with zeros (approximate as minimal distortion)
        self.D_pinhole = np.zeros((5, 1))  # 5 parameters for pinhole distortion model
        
        pinhole_params = {
            'K': self.K_pinhole.tolist(),
            'D': self.D_pinhole.tolist(),
            'DIM': dim3
        }
        
        # Save the pinhole parameters
        with open(os.path.join(self.camera_calibration_save_path,'pinhole_approximation.json'), 'w') as f:
            json.dump(pinhole_params, f, indent=2)
            
        print("\nPinhole approximation parameters:")
        print(f"K_pinhole=\n{self.K_pinhole}")
        print(f"D_pinhole=\n{self.D_pinhole}")
        
        return pinhole_params

    def undistort_to_pinhole(self, frame: np.ndarray, balance=0.0):
        """
        Undistort a frame using the pinhole approximation.
        
        Args:
            frame (np.ndarray): Input fisheye image
            balance (float): Balance between original (0.0) and zero distortion (1.0)
        
        Returns:
            np.ndarray: Undistorted image using pinhole approximation
        """
        if not hasattr(self, 'K_pinhole'):
            # Calculate pinhole parameters if not already done
            self.convert_fisheye_to_pinhole(balance=balance)
        
        h, w = frame.shape[:2]
        
        # Calculate undistortion maps if not already done
        if not hasattr(self, 'pinhole_maps'):
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D,
                np.eye(3),
                self.K_pinhole,
                (w, h),
                cv2.CV_16SC2
            )
            self.pinhole_maps = (map1, map2)
        
        # Undistort using the pinhole approximation
        undistorted = cv2.remap(
            frame,
            self.pinhole_maps[0],
            self.pinhole_maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return undistorted

    def live_streaming_comparison(self):
        """
        Start live streaming with both fisheye and pinhole undistortion for comparison.
        """
        if not self.calibrated:
            print("Camera not calibrated. Starting calibration...")
            self.calibrate_camera()
        
        self.start_streaming()
        self.convert_fisheye_to_pinhole()  # Initialize pinhole parameters
        
        print("\nStreaming controls:")
        print("'b' - Adjust balance (0.0 to 1.0)")
        print("'s' - Save current frames")
        print("'q' - Quit streaming")
        
        balance = 0.0
        frame_count = 0
        
        while self.is_running:
            frame_data = self.get_camera_frame()
            if frame_data is None:
                continue
            
            original = frame_data.rgb
            fisheye_undistorted = self.undistort_frame(original)
            pinhole_undistorted = self.undistort_to_pinhole(original, balance)
            
            # Stack images horizontally for comparison
            comparison = np.hstack([
                cv2.resize(original, (640, 480)),
                cv2.resize(fisheye_undistorted, (640, 480)),
                cv2.resize(pinhole_undistorted, (640, 480))
            ])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Fisheye Undistorted", (650, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, f"Pinhole (balance={balance:.1f})", (1290, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Comparison", comparison)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                balance = (balance + 0.1) % 1.1
                if balance > 1.0:
                    balance = 0.0
                # Recalculate pinhole parameters with new balance
                self.convert_fisheye_to_pinhole(balance=balance)
            elif key == ord('s'):
                cv2.imwrite(os.path.join(self.image_save_path,f"original_{frame_count}.jpg"), original)
                cv2.imwrite(os.path.join(self.image_save_path,f"fisheye_undistorted_{frame_count}.jpg"), fisheye_undistorted)
                cv2.imwrite(os.path.join(self.image_save_path,f"pinhole_undistorted_{frame_count}.jpg"), pinhole_undistorted)
                print(f"Saved frame set {frame_count}")
                frame_count += 1
        
        self.stop_streaming()
        cv2.destroyAllWindows()
    
    def live_streaming(self, undistort: bool = False):
        """
        Start live streaming with toggleable undistortion.
        
        Args:
            start_undistorted: Whether to start with undistortion enabled
        """
        if not self.calibrated:
            print("Camera not calibrated. Starting calibration...")
            self.calibrate_camera()
        
        self.start_streaming()
        
        print("\nStreaming controls:")
        print("'u' - Toggle undistortion")
        print("'s' - Save current frame")
        print("'q' - Quit streaming")
        
        frame_count = 0
        while self.is_running:
            frame_data = self.get_camera_frame()
            if frame_data is None:
                continue
            
            frame = frame_data.rgb
            if undistort:
                frame = self.undistort_frame(frame)
            
            # Add status text
            cv2.putText(
                frame,
                f"{'Undistorted' if undistort else 'Original'} | Press 'u' to toggle",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('u'):
                undistort = not undistort
                print(f"Undistortion: {'ON' if undistort else 'OFF'}")
            elif key == ord('s'):
                # Save both original and undistorted frames
                frame_data = self.get_camera_frame()
                if frame_data is not None:
                    original = frame_data.rgb
                    undistorted = self.undistort_frame(original)
                    cv2.imwrite(f"original_{frame_count}.jpg", original)
                    cv2.imwrite(f"undistorted_{frame_count}.jpg", undistorted)
                    print(f"Saved frame pair {frame_count}")
                    frame_count += 1
        
        self.stop_streaming()
        cv2.destroyAllWindows()

    def start_streaming(self):
        """Start the camera stream."""
        self.is_running = True
        print('started')

    def stop_streaming(self):
        """Stop the camera stream and release resources."""
        self.is_running = False
        print("stopped")

if __name__ == "__main__":
    # You can adjust the resolution and fps in this call if needed
    camera = Insta360SharedMem() # ('127.0.0.1', 8080)
    raise Exception("not here")

    cam = Insta360Calibrated(
        camera = camera, camera_resolution=(720, 720), image_save_path='images',camera_calibration_save_path='./camera_calibration'
    )
    # cam.calibrate_camera(chessboard_size=(8, 6), square_size=2.3, num_images=30)
    cam.load_calibration("camera_calibration/fisheye_calibration.json")
    cam.start_streaming()


    while True:
        read = cam.get_camera_frame()
        if read is None:
            continue 
        front, back = read.front_rgb, read.back_rgb 
        
        # cv2.imshow("Front", cam.undistort_frame(front))
        cv2.imshow("Front", front)

        cv2.imshow("Back", back)
        cv2.waitKey(1)


    # cam.start_streaming()
    # cam.live_streaming()
    # cam.live_streaming(undistort=True)  # Start with distortion by default
    cam.live_streaming_comparison()
    # cam.stop_streaming()
    cv2.destroyAllWindows()
