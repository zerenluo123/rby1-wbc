#!/usr/bin/env python3
"""Lightweight GUI to sanity-check the RBY1 GigE cameras.

This viewer wraps :class:`camera.camera_stream.AravisCameraStreamer` so we can
pick specific cameras via their serial numbers (or any other unique identifier
accepted by Aravis), stream the frames, and visualise them in simple OpenCV or
matplotlib windows.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np

# Ensure imports work whether this script is executed from the repo root
# (python scripts/...) or directly (python3 camera_stream_viewer.py).
try:  # pragma: no cover - runtime convenience
    from camera.camera_stream import AravisCameraStreamer
except ModuleNotFoundError:  # pragma: no cover - runtime convenience
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from camera.camera_stream import AravisCameraStreamer

try:  # pragma: no cover - optional dependency
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis as _Aravis
except Exception:  # pragma: no cover - headless/CI environments
    _Aravis = None


# Default serial-to-observation mapping for the five RBY1 cameras.  Override on
# the command line with ``--camera <name>=<serial>`` if the wiring differs.
DEFAULT_CAMERA_SERIAL_MAP: Mapping[str, str] = {
    "camera_head_main_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25260985",
    "camera_head_main_right_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25272263",
    "camera_head_ultrawide_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25260989",
    "camera_left_main_rgb": "FLIR-Blackfly S BFS-PGE-23S3C-24260091",
    "camera_right_main_rgb": "FLIR-Blackfly S BFS-PGE-23S3C-24260092",
}


def _parse_camera_map(entries: Sequence[str] | None) -> Dict[str, str]:
    """Convert ``KEY=SERIAL`` CLI arguments into a mapping."""

    mapping: Dict[str, str] = {}
    if not entries:
        return mapping
    for entry in entries:
        if "=" in entry:
            key, serial = entry.split("=", 1)
        else:
            # Allow ``--camera BFS_XXXX``; the window title becomes the serial.
            key, serial = entry, entry
        key = key.strip()
        serial = serial.strip()
        if not key or not serial:
            raise ValueError(f"Invalid camera specification '{entry}'")
        mapping[key] = serial
    return mapping


def _list_available_cameras() -> Sequence[str]:
    if _Aravis is None:  # pragma: no cover - requires hardware
        return []
    try:
        _Aravis.update_device_list()
        return [_Aravis.get_device_id(i) for i in range(_Aravis.get_n_devices())]
    except Exception:
        return []


def _print_startup_help(camera_map: Mapping[str, str], exc: Exception) -> None:
    print(f"[camera] Failed to start camera streams: {exc}", file=sys.stderr)
    available = _list_available_cameras()
    if camera_map:
        print("[camera] Requested mapping:", file=sys.stderr)
        for key, serial in camera_map.items():
            print(f"  {key}: {serial}", file=sys.stderr)
    if available:
        print("[camera] Detected devices (use these identifiers with --camera):", file=sys.stderr)
        for dev in available:
            print(f"  {dev}", file=sys.stderr)
    else:
        print(
            "[camera] No Aravis devices detected. Ensure the cameras are powered and connected.",
            file=sys.stderr,
        )


def _compute_display_scale(height: int, base_scale: float | None, max_height: int | None) -> float:
    """Determine the display scale for an image with the given height."""

    scale = base_scale if base_scale is not None else 1.0
    if max_height is not None and max_height > 0 and height > max_height:
        auto_scale = max_height / float(height)
        scale = min(scale, auto_scale) if base_scale is not None else auto_scale
    return max(scale, 1e-3)


def _resize_frame(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize frames for display while handling environments without OpenCV."""

    scale = float(scale)
    if abs(scale - 1.0) < 1e-3:
        return image

    height, width = image.shape[:2]
    new_h = max(int(round(height * scale)), 1)
    new_w = max(int(round(width * scale)), 1)
    if new_h == height and new_w == width:
        return image

    try:  # pragma: no cover - optional dependency
        import cv2

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    except Exception:
        rows = np.linspace(0, height - 1, new_h, dtype=int)
        cols = np.linspace(0, width - 1, new_w, dtype=int)
        return image[np.ix_(rows, cols)]


def _install_sigint_handler(stop_flag: MutableMapping[str, bool]) -> None:
    """Gracefully exit on Ctrl+C by flipping a shared stop flag."""

    def _handler(signum: int, frame: object | None) -> None:  # pragma: no cover - signal handler
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class _BaseViewer:
    """Common interface for GUI backends."""

    def update(self, frames: Mapping[str, np.ndarray]) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _OpenCVViewer(_BaseViewer):
    def __init__(
        self,
        refresh_hz: float,
        *,
        scale: float | None,
        max_height: int | None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenCV is not available. Install opencv-python to use this backend.") from exc

        self._cv2 = cv2
        self._base_scale = scale
        self._max_height = max_height
        self._wait_ms = max(int(1000.0 / max(refresh_hz, 1e-3)), 1)

    def update(self, frames: Mapping[str, np.ndarray]) -> bool:
        for window, rgb_image in frames.items():
            image = rgb_image
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            scale = _compute_display_scale(image.shape[0], self._base_scale, self._max_height)
            image = _resize_frame(image, scale)
            # bgr = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)
            self._cv2.imshow(window, image)
        key = self._cv2.waitKey(self._wait_ms) & 0xFF
        return key not in (27, ord("q"))  # ESC or q closes the viewer

    def close(self) -> None:
        self._cv2.destroyAllWindows()


class _MatplotlibViewer(_BaseViewer):
    def __init__(
        self,
        camera_keys: Sequence[str],
        refresh_hz: float,
        *,
        scale: float | None,
        max_height: int | None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "matplotlib is not available. Install matplotlib or choose the OpenCV backend."
            ) from exc

        self._plt = plt
        self._plt.ion()
        n = len(camera_keys)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        self._fig, axes = self._plt.subplots(rows, cols, squeeze=False)
        self._images: Dict[str, object] = {}
        self._axes = axes
        self._camera_keys = list(camera_keys)
        self._pause_dt = 1.0 / max(refresh_hz, 1e-3)
        self._base_scale = scale
        self._max_height = max_height
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if idx < n:
                    key = self._camera_keys[idx]
                    img = ax.imshow(blank)
                    ax.set_title(key)
                    ax.axis("off")
                    self._images[key] = img
                    idx += 1
                else:
                    ax.axis("off")

    def update(self, frames: Mapping[str, np.ndarray]) -> bool:
        for key, image in frames.items():
            if key not in self._images:
                continue
            scale = _compute_display_scale(image.shape[0], self._base_scale, self._max_height)
            image = _resize_frame(image, scale)
            self._images[key].set_data(image)
        self._fig.canvas.draw_idle()
        self._plt.pause(self._pause_dt)
        return self._plt.fignum_exists(self._fig.number)

    def close(self) -> None:
        self._plt.close(self._fig)


def _build_viewer(
    backend: str,
    camera_keys: Sequence[str],
    refresh_hz: float,
    *,
    scale: float | None,
    max_height: int | None,
) -> _BaseViewer:
    backend = backend.lower()
    if backend == "opencv":
        return _OpenCVViewer(refresh_hz=refresh_hz, scale=scale, max_height=max_height)
    if backend == "matplotlib":
        return _MatplotlibViewer(
            camera_keys=camera_keys,
            refresh_hz=refresh_hz,
            scale=scale,
            max_height=max_height,
        )
    try:
        return _OpenCVViewer(refresh_hz=refresh_hz, scale=scale, max_height=max_height)
    except RuntimeError:
        return _MatplotlibViewer(
            camera_keys=camera_keys,
            refresh_hz=refresh_hz,
            scale=scale,
            max_height=max_height,
        )


def _log_camera_shapes(streamer: AravisCameraStreamer) -> None:
    try:
        window = streamer.get_observation_window(horizon=1)
    except Exception as exc:
        print(f"[camera] Unable to inspect frame shapes: {exc}", file=sys.stderr)
        return
    for key, frames in window.items():
        frame = frames[-1]
        height, width = frame.shape[:2]
        print(f"[camera] {key}: {width}x{height}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw frames from the RBY1 cameras")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=SERIAL",
        help=(
            "Camera mapping in the form observation_key=serial. Repeat for each camera. "
            "If omitted the default RBY1 mapping is used. "
            "Serial numbers can be obtained from `arv-tool-0.8 list`."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "matplotlib"),
        default="auto",
        help="GUI backend (default: auto-detect, preferring OpenCV).",
    )
    parser.add_argument("--buffer-size", type=int, default=16, help="Per-camera frame buffer size.")
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=15.0,
        help="How often to update the GUI windows (in Hz).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Fixed display scaling factor. Leave unset to auto-scale based on --max-height.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=800,
        help="Automatically downscale frames so their height does not exceed this value (0 disables).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic frames instead of using hardware (for testing/headless environments).",
    )
    parser.add_argument(
        "--mock-resolution",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(720, 1280),
        help="Resolution of the synthetic frames when --mock is enabled.",
    )
    parser.add_argument(
        "--init-timeout",
        type=float,
        default=4.0,
        help="Seconds to wait for buffers to fill before rendering.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of frames per camera to fetch per refresh (default: 1).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered Aravis devices and exit (no streaming).",
    )
    args = parser.parse_args()

    if args.list:
        detected = _list_available_cameras()
        if not detected:
            print("No Aravis devices detected. Ensure the cameras are connected and powered.")
        else:
            print("Detected cameras:")
            for dev in detected:
                print(f"  {dev}")
        return

    camera_map = _parse_camera_map(args.camera)
    if not camera_map:
        camera_map = dict(DEFAULT_CAMERA_SERIAL_MAP)
        if not camera_map:
            parser.error("No cameras specified; pass --camera NAME=SERIAL to stream a device.")

    camera_streamer = AravisCameraStreamer(
        camera_map,
        buffer_size=max(args.buffer_size, 1),
        mock_mode=args.mock,
        mock_resolution=tuple(args.mock_resolution),
    )

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    try:
        camera_streamer.start()
    except Exception as exc:
        _print_startup_help(camera_map, exc)
        raise
    try:
        camera_streamer.wait_until_ready(min_frames=1, timeout=max(args.init_timeout, 0.0))
    except TimeoutError as exc:
        print(f"[camera] Warning: {exc}. Continuing anyway.", file=sys.stderr)

    _log_camera_shapes(camera_streamer)

    viewer: _BaseViewer | None = None
    viewer = _build_viewer(
        backend=args.backend,
        camera_keys=list(camera_map.keys()),
        refresh_hz=args.refresh_hz,
        scale=args.scale,
        max_height=args.max_height,
    )

    refresh_dt = 1.0 / max(args.refresh_hz, 1e-3)
    try:
        while not stop_flag["stop"]:
            try:
                window = camera_streamer.get_observation_window(horizon=max(args.horizon, 1))
            except RuntimeError as exc:
                print(f"[camera] {exc}", file=sys.stderr)
                time.sleep(refresh_dt)
                continue
            frames = {k: v[-1] for k, v in window.items()}
            if not viewer.update(frames):
                break
            time.sleep(refresh_dt)
    finally:
        if viewer is not None:
            viewer.close()
        camera_streamer.stop()


if __name__ == "__main__":
    main()
