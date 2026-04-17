from __future__ import annotations

import math
from dataclasses import dataclass, field

from .detectors import Detection


@dataclass
class TrackPoint:
    frame_idx: int
    x_px: float
    y_px: float
    confidence: float


@dataclass
class Track:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]


@dataclass
class NearestNeighborTracker:
    """Simple multi-object tracker for microscopy particles.

    This is not as robust as SORT/DeepSORT, but it gives a clean baseline and
    preserves the same output shape you would want from a stronger tracker.
    """

    max_distance_px: float = 18.0
    max_missed_frames: int = 4
    next_track_id: int = 1
    active_tracks: list[Track] = field(default_factory=list)
    finished_tracks: list[Track] = field(default_factory=list)

    def update(self, frame_idx: int, detections: list[Detection]) -> None:
        unmatched_detection_indices = set(range(len(detections)))
        assignments: list[tuple[int, int]] = []

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.active_tracks):
            last = track.last_point
            for detection_idx, detection in enumerate(detections):
                distance = math.dist((last.x_px, last.y_px), (detection.x_px, detection.y_px))
                if distance <= self.max_distance_px:
                    candidate_pairs.append((distance, track_idx, detection_idx))

        matched_tracks: set[int] = set()
        for _distance, track_idx, detection_idx in sorted(candidate_pairs):
            if track_idx in matched_tracks or detection_idx not in unmatched_detection_indices:
                continue
            assignments.append((track_idx, detection_idx))
            matched_tracks.add(track_idx)
            unmatched_detection_indices.remove(detection_idx)

        next_active_tracks: list[Track] = []

        for track_idx, track in enumerate(self.active_tracks):
            if track_idx in matched_tracks:
                detection_idx = next(
                    assigned_detection_idx
                    for assigned_track_idx, assigned_detection_idx in assignments
                    if assigned_track_idx == track_idx
                )
                detection = detections[detection_idx]
                track.points.append(
                    TrackPoint(
                        frame_idx=frame_idx,
                        x_px=detection.x_px,
                        y_px=detection.y_px,
                        confidence=detection.confidence,
                    )
                )
                track.missed_frames = 0
                next_active_tracks.append(track)
                continue

            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                self.finished_tracks.append(track)
            else:
                next_active_tracks.append(track)

        for detection_idx in sorted(unmatched_detection_indices):
            detection = detections[detection_idx]
            next_active_tracks.append(
                Track(
                    track_id=self.next_track_id,
                    points=[
                        TrackPoint(
                            frame_idx=frame_idx,
                            x_px=detection.x_px,
                            y_px=detection.y_px,
                            confidence=detection.confidence,
                        )
                    ],
                )
            )
            self.next_track_id += 1

        self.active_tracks = next_active_tracks

    def finalize(self) -> list[Track]:
        finished = self.finished_tracks + self.active_tracks
        self.finished_tracks = []
        self.active_tracks = []
        return sorted(finished, key=lambda track: track.track_id)

