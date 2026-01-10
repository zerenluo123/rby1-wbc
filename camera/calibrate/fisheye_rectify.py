import cv2
import numpy as np
import json

data = json.load(open("calib_fisheye_out/calibration_fisheye.json", "r"))
K = np.array(data["K"], dtype=np.float64)
D = np.array(data["D"], dtype=np.float64)
W = data["image_size"]["width"]
H = data["image_size"]["height"]

# Choose output camera matrix.
# balance=0 gives a tighter crop; balance=1 keeps more FOV (more black borders).
balance = 0.0
newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (W, H), np.eye(3), balance=balance, new_size=(W, H)
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), newK, (W, H), m1type=cv2.CV_16SC2
)

# For each frame 'img' (BGR):
# undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
