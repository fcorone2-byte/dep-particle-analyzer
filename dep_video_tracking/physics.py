from __future__ import annotations

import math
from dataclasses import dataclass

from .tracking import Track


@dataclass(frozen=True)
class FrequencySchedule:
    start_khz: float
    stop_khz: float
    num_frames: int

    def frequency_at_frame(self, frame_idx: int) -> float:
        if self.num_frames <= 1:
            return self.start_khz
        fraction = frame_idx / float(self.num_frames - 1)
        return self.start_khz + fraction * (self.stop_khz - self.start_khz)


@dataclass(frozen=True)
class TrackMetrics:
    track_id: int
    num_points: int
    mean_speed_um_s: float
    max_speed_um_s: float
    net_displacement_um: float
    crossover_frame: int | None
    crossover_time_s: float | None
    crossover_frequency_khz: float | None
    final_region: str
    final_x_px: float
    final_y_px: float


def classify_track_endpoint(
    track: Track,
    *,
    outlet_x_threshold: float | None,
    outlet_center_y: float | None,
    outlet_center_half_height: float,
    outlet_inner_half_height: float,
) -> tuple[str, float, float]:
    endpoint = track.points[-1]
    x_px = endpoint.x_px
    y_px = endpoint.y_px

    if outlet_x_threshold is None or outlet_center_y is None:
        return "Unclassified", x_px, y_px

    if x_px < outlet_x_threshold:
        return "Inlet", x_px, y_px

    dy = y_px - outlet_center_y
    if abs(dy) <= outlet_center_half_height:
        return "C", x_px, y_px
    if dy < 0:
        return ("S2T" if abs(dy) <= outlet_inner_half_height else "S1T"), x_px, y_px
    return ("S2B" if abs(dy) <= outlet_inner_half_height else "S1B"), x_px, y_px


def instantaneous_speeds_um_s(track: Track, pixel_size_um: float, fps: float) -> list[float]:
    if len(track.points) < 2:
        return []

    speeds: list[float] = []
    for previous, current in zip(track.points[:-1], track.points[1:]):
        frame_delta = current.frame_idx - previous.frame_idx
        if frame_delta <= 0:
            continue
        distance_px = math.dist((previous.x_px, previous.y_px), (current.x_px, current.y_px))
        distance_um = distance_px * pixel_size_um
        time_s = frame_delta / fps
        speeds.append(distance_um / time_s)
    return speeds


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        section = values[start : idx + 1]
        result.append(sum(section) / len(section))
    return result


def detect_crossover(
    track: Track,
    *,
    pixel_size_um: float,
    fps: float,
    velocity_threshold_um_s: float,
    stall_frames: int,
    rolling_window: int,
    frequency_schedule: FrequencySchedule | None,
) -> tuple[int | None, float | None, float | None]:
    speeds = instantaneous_speeds_um_s(track, pixel_size_um=pixel_size_um, fps=fps)
    if not speeds:
        return None, None, None

    smoothed = rolling_mean(speeds, rolling_window)
    consecutive = 0
    for speed_idx, speed in enumerate(smoothed):
        if speed <= velocity_threshold_um_s:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= stall_frames:
            point_idx = min(speed_idx + 1, len(track.points) - 1)
            frame_idx = track.points[point_idx].frame_idx
            time_s = frame_idx / fps
            frequency_khz = (
                frequency_schedule.frequency_at_frame(frame_idx)
                if frequency_schedule is not None
                else None
            )
            return frame_idx, time_s, frequency_khz

    return None, None, None


def summarize_track(
    track: Track,
    *,
    pixel_size_um: float,
    fps: float,
    velocity_threshold_um_s: float,
    stall_frames: int,
    rolling_window: int,
    frequency_schedule: FrequencySchedule | None,
    outlet_x_threshold: float | None,
    outlet_center_y: float | None,
    outlet_center_half_height: float,
    outlet_inner_half_height: float,
) -> TrackMetrics:
    speeds = instantaneous_speeds_um_s(track, pixel_size_um=pixel_size_um, fps=fps)
    if len(track.points) >= 2:
        start = track.points[0]
        end = track.points[-1]
        net_displacement_um = math.dist((start.x_px, start.y_px), (end.x_px, end.y_px)) * pixel_size_um
    else:
        net_displacement_um = 0.0

    crossover_frame, crossover_time_s, crossover_frequency_khz = detect_crossover(
        track,
        pixel_size_um=pixel_size_um,
        fps=fps,
        velocity_threshold_um_s=velocity_threshold_um_s,
        stall_frames=stall_frames,
        rolling_window=rolling_window,
        frequency_schedule=frequency_schedule,
    )
    final_region, final_x_px, final_y_px = classify_track_endpoint(
        track,
        outlet_x_threshold=outlet_x_threshold,
        outlet_center_y=outlet_center_y,
        outlet_center_half_height=outlet_center_half_height,
        outlet_inner_half_height=outlet_inner_half_height,
    )

    return TrackMetrics(
        track_id=track.track_id,
        num_points=len(track.points),
        mean_speed_um_s=sum(speeds) / len(speeds) if speeds else 0.0,
        max_speed_um_s=max(speeds) if speeds else 0.0,
        net_displacement_um=net_displacement_um,
        crossover_frame=crossover_frame,
        crossover_time_s=crossover_time_s,
        crossover_frequency_khz=crossover_frequency_khz,
        final_region=final_region,
        final_x_px=final_x_px,
        final_y_px=final_y_px,
    )
