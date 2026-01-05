#!/usr/bin/env python3
"""Compose saved debug images into a tiled video for one inference batch."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

BATCH_PATTERN = re.compile(r"^(?P<batch>.+_\d{3})_(?P<camera>.+)_frame_(?P<frame>\d+)\.png$")
BATCH_SORT_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<seq>\d+)$")


class ImageEntry:
    def __init__(self, batch: str, camera: str, frame: int, path: Path) -> None:
        self.batch = batch
        self.camera = camera
        self.frame = frame
        self.path = path


def _find_image_dir(root: Path) -> Path:
    if any(root.glob("*.png")):
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*.png"))]
    if not candidates:
        raise FileNotFoundError(f"No png images found under {root}")
    return sorted(candidates)[-1]


def _infer_batch_tag(paths: Iterable[Path]) -> str:
    counts: Dict[str, int] = {}
    for path in paths:
        match = BATCH_PATTERN.match(path.name)
        if not match:
            continue
        batch = match.group("batch")
        counts[batch] = counts.get(batch, 0) + 1
    if not counts:
        raise ValueError("Unable to infer batch tag; pass --batch-tag explicitly.")
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _collect_entries(paths: Iterable[Path], batch_tag: Optional[str]) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    for path in paths:
        match = BATCH_PATTERN.match(path.name)
        if not match:
            continue
        batch = match.group("batch")
        if batch_tag is not None and batch != batch_tag:
            continue
        entries.append(
            ImageEntry(
                batch=batch,
                camera=match.group("camera"),
                frame=int(match.group("frame")),
                path=path,
            )
        )
    if not entries:
        raise ValueError(
            "No images found for the requested batch tag." if batch_tag else "No images found."
        )
    return entries


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def _overlay_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _resize_with_letterbox(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _batch_sort_key(batch: str) -> Tuple[str, int]:
    match = BATCH_SORT_PATTERN.match(batch)
    if match:
        timestamp = match.group("date") + match.group("time")
        return (timestamp, int(match.group("seq")))
    return (batch, 0)


def _build_timeline(
    entries: List[ImageEntry],
) -> Tuple[List[str], List[Tuple[str, int]], Dict[Tuple[str, int], Dict[str, Path]]]:
    cameras = sorted({entry.camera for entry in entries})
    timeline_map: Dict[Tuple[str, int], Dict[str, Path]] = {}
    timeline_keys: List[Tuple[str, int]] = []
    seen_keys = set()

    def _entry_sort_key(entry: ImageEntry) -> Tuple[Tuple[str, int], int]:
        return (_batch_sort_key(entry.batch), entry.frame)

    for entry in sorted(entries, key=_entry_sort_key):
        key = (entry.batch, entry.frame)
        timeline_map.setdefault(key, {})[entry.camera] = entry.path
        if key not in seen_keys:
            timeline_keys.append(key)
            seen_keys.add(key)

    return cameras, timeline_keys, timeline_map


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _infer_tile_size(
    cameras: List[str],
    timeline_keys: List[Tuple[str, int]],
    timeline_map: Dict[Tuple[str, int], Dict[str, Path]],
) -> Tuple[int, int]:
    sample_images = []
    for camera in cameras:
        sample_path = None
        for key in timeline_keys:
            sample_path = timeline_map.get(key, {}).get(camera)
            if sample_path is not None:
                break
        if sample_path is not None:
            sample_images.append(_read_image(sample_path))
    if not sample_images:
        raise ValueError("Unable to read any images to determine tile size.")
    tile_h = max(img.shape[0] for img in sample_images)
    tile_w = max(img.shape[1] for img in sample_images)
    return _ensure_even(tile_w), _ensure_even(tile_h)


def _compose_video(
    cameras: List[str],
    timeline_keys: List[Tuple[str, int]],
    timeline_map: Dict[Tuple[str, int], Dict[str, Path]],
    output_path: Path,
    fps: float,
    label_cameras: bool,
    codec: str,
) -> None:
    if not timeline_keys:
        raise ValueError("No frames found across cameras.")

    tile_w, tile_h = _infer_tile_size(cameras, timeline_keys, timeline_map)

    cols = int(math.ceil(math.sqrt(len(cameras))))
    rows = int(math.ceil(len(cameras) / cols))
    frame_size = (cols * tile_w, rows * tile_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for key in timeline_keys:
            canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
            for cam_idx, camera in enumerate(cameras):
                r = cam_idx // cols
                c = cam_idx % cols
                image_path = timeline_map.get(key, {}).get(camera)
                if image_path is None:
                    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                else:
                    tile = _read_image(image_path)
                    tile = _resize_with_letterbox(tile, tile_w, tile_h)
                if label_cameras:
                    tile = _overlay_label(tile, camera)
                y0 = r * tile_h
                x0 = c * tile_w
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile per-camera debug images from one inference batch into a video."
    )
    parser.add_argument("input_dir", help="Directory with debug images or its parent directory.")
    parser.add_argument("--batch-tag", default=None, help="Batch tag prefix in the filenames.")
    parser.add_argument("--output", default=None, help="Output video path (mp4).")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for the output video.")
    parser.add_argument(
        "--label-cameras",
        action="store_true",
        help="Overlay camera names on each tile.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec to use (default: mp4v).",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    image_dir = _find_image_dir(root)
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No png images found under {image_dir}")

    if args.batch_tag is None:
        entries = _collect_entries(paths, batch_tag=None)
    else:
        entries = _collect_entries(paths, batch_tag=args.batch_tag)

    cameras, timeline_keys, timeline_map = _build_timeline(entries)
    batch_count = len({batch for batch, _ in timeline_keys})
    print(f"[debug] Cameras: {len(cameras)} | Batches: {batch_count} | Frames: {len(timeline_keys)}")

    output_path = Path(args.output) if args.output else image_dir / "debug_grid.mp4"
    _compose_video(
        cameras,
        timeline_keys,
        timeline_map,
        output_path,
        fps=args.fps,
        label_cameras=args.label_cameras,
        codec=args.codec,
    )
    print(f"Wrote video to {output_path}")


if __name__ == "__main__":
    main()
