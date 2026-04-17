from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    x_px: float
    y_px: float
    area_px: int
    confidence: float = 1.0


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return particle detections for a single frame."""


@dataclass
class ThresholdBlobDetector:
    """Simple fallback detector for bright-field style microscopy frames.

    This is intentionally lightweight so the full pipeline can run without a
    heavy ML dependency. Later, a YOLO or CellPose detector can implement the
    same `detect()` interface.
    """

    color_spread_min: int = 18
    color_max_channel: int = 240
    min_component_area: int = 4
    max_component_area: int = 400

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Expected an RGB frame with shape (H, W, 3).")

        rgb = frame[:, :, :3]
        channel_max = rgb.max(axis=2)
        channel_min = rgb.min(axis=2)
        mask = (channel_max - channel_min >= self.color_spread_min) & (
            channel_max <= self.color_max_channel
        )

        return connected_component_centroids(
            mask=mask,
            min_component_area=self.min_component_area,
            max_component_area=self.max_component_area,
        )


class YoloDetector:
    """Optional detector adapter for Ultralytics YOLO models."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "Ultralytics is not installed. Install `ultralytics` to use "
                "the YOLO detector backend."
            ) from exc

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.confidence_threshold,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                x_px = (x1 + x2) / 2.0
                y_px = (y1 + y2) / 2.0
                area_px = int(max(1.0, (x2 - x1) * (y2 - y1)))
                detections.append(
                    Detection(
                        x_px=x_px,
                        y_px=y_px,
                        area_px=area_px,
                        confidence=confidence,
                    )
                )
        return detections


def connected_component_centroids(
    *,
    mask: np.ndarray,
    min_component_area: int,
    max_component_area: int,
) -> list[Detection]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    detections: list[Detection] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue

            queue = deque([(y, x)])
            seen[y, x] = True
            points: list[tuple[int, int]] = []

            while queue:
                cy, cx = queue.popleft()
                points.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))

            area = len(points)
            if min_component_area <= area <= max_component_area:
                xs = [point[1] for point in points]
                ys = [point[0] for point in points]
                detections.append(
                    Detection(
                        x_px=float(np.mean(xs)),
                        y_px=float(np.mean(ys)),
                        area_px=area,
                    )
                )

    return detections

