#!/usr/bin/env python3
"""Detect, track, and quantify particle motion from a video or image stack."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dep_video_tracking.detectors import Detection, ThresholdBlobDetector, YoloDetector
from dep_video_tracking.physics import FrequencySchedule, rolling_mean, summarize_track
from dep_video_tracking.tracking import NearestNeighborTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track microscopy particles and estimate velocities plus crossover frequency."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video or multi-frame image.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and PNG outputs.")
    parser.add_argument("--fps", type=float, required=True, help="Frames per second.")
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        required=True,
        help="Microscope calibration in micrometers per pixel.",
    )
    parser.add_argument(
        "--detector",
        choices=["blob", "yolo"],
        default="blob",
        help="Detection backend.",
    )
    parser.add_argument("--yolo-model", type=str, help="YOLO model path when --detector yolo is used.")
    parser.add_argument("--max-distance-px", type=float, default=18.0, help="Tracker match radius.")
    parser.add_argument("--max-missed-frames", type=int, default=4, help="Frames a track can disappear.")
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=4,
        help="Discard tracks shorter than this many detections.",
    )
    parser.add_argument(
        "--velocity-threshold-um-s",
        type=float,
        default=0.25,
        help="Stall threshold for crossover detection.",
    )
    parser.add_argument(
        "--stall-frames",
        type=int,
        default=3,
        help="Consecutive low-speed steps needed to call crossover.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=3,
        help="Window size for smoothing instantaneous speed.",
    )
    parser.add_argument("--start-frequency-khz", type=float, help="Applied frequency at frame 0.")
    parser.add_argument("--stop-frequency-khz", type=float, help="Applied frequency at the final frame.")
    parser.add_argument(
        "--blob-color-spread-min",
        type=int,
        default=18,
        help="Blob detector color spread threshold.",
    )
    parser.add_argument(
        "--blob-color-max-channel",
        type=int,
        default=240,
        help="Blob detector rejects brighter pixels above this channel value.",
    )
    parser.add_argument("--blob-min-area", type=int, default=4, help="Blob detector minimum area.")
    parser.add_argument("--blob-max-area", type=int, default=400, help="Blob detector maximum area.")
    parser.add_argument("--roi-x-min", type=int, default=0, help="Left crop boundary in pixels.")
    parser.add_argument("--roi-x-max", type=int, help="Right crop boundary in pixels.")
    parser.add_argument("--roi-y-min", type=int, default=0, help="Top crop boundary in pixels.")
    parser.add_argument("--roi-y-max", type=int, help="Bottom crop boundary in pixels.")
    parser.add_argument(
        "--outlet-x-threshold",
        type=float,
        help="Particles ending left of this x-position are labeled Inlet instead of outlet lanes.",
    )
    parser.add_argument(
        "--outlet-center-y",
        type=float,
        help="Center y-position of the middle outlet lane.",
    )
    parser.add_argument(
        "--outlet-center-half-height",
        type=float,
        default=35.0,
        help="Half-height of the center outlet lane in pixels.",
    )
    parser.add_argument(
        "--outlet-inner-half-height",
        type=float,
        default=68.0,
        help="Boundary between the inner S2 lanes and outer S1 lanes.",
    )
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    if args.detector == "yolo":
        if not args.yolo_model:
            raise ValueError("--yolo-model is required when --detector yolo is used.")
        return YoloDetector(model_path=args.yolo_model)

    return ThresholdBlobDetector(
        color_spread_min=args.blob_color_spread_min,
        color_max_channel=args.blob_color_max_channel,
        min_component_area=args.blob_min_area,
        max_component_area=args.blob_max_area,
    )


def load_frames(input_path: Path) -> list[np.ndarray]:
    if input_path.is_dir():
        allowed_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
        frame_paths = sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
        if not frame_paths:
            raise ValueError(f"No image frames found in directory: {input_path}")
        return [np.array(Image.open(path).convert("RGB")) for path in frame_paths]

    try:
        import imageio.v2 as iio
    except ImportError:
        iio = None

    if iio is not None:
        reader = iio.get_reader(str(input_path))
        try:
            return [frame for frame in reader]
        finally:
            reader.close()

    suffix = input_path.suffix.lower()
    if suffix not in {".gif", ".tif", ".tiff"}:
        raise ImportError(
            "Loading video files requires `imageio` in this environment. "
            "Install `imageio` for MP4/AVI/WEBM input, or use a GIF/TIFF stack."
        )

    image = Image.open(input_path)
    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        try:
            image.seek(frame_index)
        except EOFError:
            break
        frames.append(np.array(image.convert("RGB")))
        frame_index += 1
    return frames


def resolve_roi(
    frame_shape: tuple[int, int, int],
    *,
    x_min: int,
    x_max: int | None,
    y_min: int,
    y_max: int | None,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x0 = max(0, x_min)
    y0 = max(0, y_min)
    x1 = width if x_max is None else min(width, x_max)
    y1 = height if y_max is None else min(height, y_max)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Invalid ROI after clipping: {(x0, x1, y0, y1)}")
    return x0, x1, y0, y1


def classify_outlet(
    x_px: float,
    y_px: float,
    *,
    x_threshold: float | None,
    center_y: float | None,
    center_half_height: float,
    inner_half_height: float,
) -> str:
    if x_threshold is None or center_y is None:
        return "Unclassified"

    if x_px < x_threshold:
        return "Inlet"

    dy = y_px - center_y
    if abs(dy) <= center_half_height:
        return "C"
    if dy < 0:
        return "S2T" if abs(dy) <= inner_half_height else "S1T"
    return "S2B" if abs(dy) <= inner_half_height else "S1B"


def write_trajectory_csv(
    output_path: Path,
    tracks,
    *,
    fps: float,
    pixel_size_um: float,
    frequency_schedule: FrequencySchedule | None,
    rolling_window: int,
) -> None:
    fieldnames = [
        "track_id",
        "frame_idx",
        "time_s",
        "x_px",
        "y_px",
        "x_um",
        "y_um",
        "instantaneous_speed_um_s",
        "rolling_speed_um_s",
        "frequency_khz",
        "confidence",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for track in tracks:
            instantaneous_speeds: list[float | None] = [None]
            for previous, current in zip(track.points[:-1], track.points[1:]):
                frame_delta = current.frame_idx - previous.frame_idx
                if frame_delta <= 0:
                    instantaneous_speeds.append(None)
                    continue
                distance_px = ((current.x_px - previous.x_px) ** 2 + (current.y_px - previous.y_px) ** 2) ** 0.5
                distance_um = distance_px * pixel_size_um
                instantaneous_speeds.append(distance_um / (frame_delta / fps))

            smoothed = rolling_mean(
                [speed if speed is not None else 0.0 for speed in instantaneous_speeds[1:]],
                rolling_window,
            )
            smoothed = [None] + smoothed

            for point_idx, point in enumerate(track.points):
                frame_idx = point.frame_idx
                writer.writerow(
                    {
                        "track_id": track.track_id,
                        "frame_idx": frame_idx,
                        "time_s": round(frame_idx / fps, 6),
                        "x_px": round(point.x_px, 4),
                        "y_px": round(point.y_px, 4),
                        "x_um": round(point.x_px * pixel_size_um, 4),
                        "y_um": round(point.y_px * pixel_size_um, 4),
                        "instantaneous_speed_um_s": (
                            "" if instantaneous_speeds[point_idx] is None else round(instantaneous_speeds[point_idx], 6)
                        ),
                        "rolling_speed_um_s": (
                            "" if smoothed[point_idx] is None else round(smoothed[point_idx], 6)
                        ),
                        "frequency_khz": (
                            "" if frequency_schedule is None else round(frequency_schedule.frequency_at_frame(frame_idx), 6)
                        ),
                        "confidence": round(point.confidence, 4),
                    }
                )


def write_summary_csv(output_path: Path, metrics) -> None:
    fieldnames = [
        "track_id",
        "num_points",
        "mean_speed_um_s",
        "max_speed_um_s",
        "net_displacement_um",
        "crossover_frame",
        "crossover_time_s",
        "crossover_frequency_khz",
        "final_region",
        "final_x_px",
        "final_y_px",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "track_id": metric.track_id,
                    "num_points": metric.num_points,
                    "mean_speed_um_s": round(metric.mean_speed_um_s, 6),
                    "max_speed_um_s": round(metric.max_speed_um_s, 6),
                    "net_displacement_um": round(metric.net_displacement_um, 6),
                    "crossover_frame": "" if metric.crossover_frame is None else metric.crossover_frame,
                    "crossover_time_s": "" if metric.crossover_time_s is None else round(metric.crossover_time_s, 6),
                    "crossover_frequency_khz": (
                        "" if metric.crossover_frequency_khz is None else round(metric.crossover_frequency_khz, 6)
                    ),
                    "final_region": metric.final_region,
                    "final_x_px": round(metric.final_x_px, 4),
                    "final_y_px": round(metric.final_y_px, 4),
                }
            )


def write_region_counts_csv(output_path: Path, counts: dict[str, int]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region", "count"])
        writer.writeheader()
        for region, count in counts.items():
            writer.writerow({"region": region, "count": count})


def write_preview(
    output_path: Path,
    frame: np.ndarray,
    tracks,
    *,
    roi: tuple[int, int, int, int] | None = None,
    outlet_x_threshold: float | None = None,
    outlet_center_y: float | None = None,
    outlet_center_half_height: float | None = None,
    outlet_inner_half_height: float | None = None,
) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    if roi is not None:
        x0, x1, y0, y1 = roi
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 120, 255), width=2)
    if outlet_x_threshold is not None:
        draw.line(
            ((outlet_x_threshold, 0), (outlet_x_threshold, frame.shape[0] - 1)),
            fill=(255, 165, 0),
            width=2,
        )
        draw.text((outlet_x_threshold + 6, 8), "Outlet boundary", fill=(255, 165, 0))
    if (
        outlet_center_y is not None
        and outlet_center_half_height is not None
        and outlet_inner_half_height is not None
    ):
        y_lines = [
            (outlet_center_y, "C center", (0, 180, 0)),
            (outlet_center_y - outlet_center_half_height, "C top", (0, 180, 0)),
            (outlet_center_y + outlet_center_half_height, "C bottom", (0, 180, 0)),
            (outlet_center_y - outlet_inner_half_height, "S2T/S1T", (180, 0, 180)),
            (outlet_center_y + outlet_inner_half_height, "S2B/S1B", (180, 0, 180)),
        ]
        for y_value, label, color in y_lines:
            draw.line(((0, y_value), (frame.shape[1] - 1, y_value)), fill=color, width=2)
            draw.text((8, y_value + 4), label, fill=color)
    for track in tracks:
        points = [(point.x_px, point.y_px) for point in track.points]
        if len(points) >= 2:
            draw.line(points, fill=(255, 0, 0), width=2)
        if points:
            x_px, y_px = points[-1]
            r = 6
            draw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), outline=(0, 200, 0), width=2)
            draw.text((x_px + 8, y_px - 8), f"#{track.track_id:03d}", fill=(0, 0, 0))
    image.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.input)
    roi = resolve_roi(
        frames[0].shape,
        x_min=args.roi_x_min,
        x_max=args.roi_x_max,
        y_min=args.roi_y_min,
        y_max=args.roi_y_max,
    )
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = roi
    detector = build_detector(args)
    tracker = NearestNeighborTracker(
        max_distance_px=args.max_distance_px,
        max_missed_frames=args.max_missed_frames,
    )

    for frame_idx, frame in enumerate(frames):
        cropped_frame = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        detections = [
            Detection(
                x_px=detection.x_px + roi_x_min,
                y_px=detection.y_px + roi_y_min,
                area_px=detection.area_px,
                confidence=detection.confidence,
            )
            for detection in detector.detect(cropped_frame)
        ]
        tracker.update(frame_idx=frame_idx, detections=detections)

    all_tracks = tracker.finalize()
    tracks = [track for track in all_tracks if len(track.points) >= args.min_track_length]

    frequency_schedule = None
    if args.start_frequency_khz is not None and args.stop_frequency_khz is not None:
        frequency_schedule = FrequencySchedule(
            start_khz=args.start_frequency_khz,
            stop_khz=args.stop_frequency_khz,
            num_frames=len(frames),
        )

    metrics = [
        summarize_track(
            track,
            pixel_size_um=args.pixel_size_um,
            fps=args.fps,
            velocity_threshold_um_s=args.velocity_threshold_um_s,
            stall_frames=args.stall_frames,
            rolling_window=args.rolling_window,
            frequency_schedule=frequency_schedule,
            outlet_x_threshold=args.outlet_x_threshold,
            outlet_center_y=args.outlet_center_y,
            outlet_center_half_height=args.outlet_center_half_height,
            outlet_inner_half_height=args.outlet_inner_half_height,
        )
        for track in tracks
    ]

    region_order = ["S1T", "S2T", "C", "S2B", "S1B", "Inlet", "Unclassified"]
    region_counts = {region: 0 for region in region_order}
    for metric in metrics:
        region_counts.setdefault(metric.final_region, 0)
        region_counts[metric.final_region] += 1

    write_trajectory_csv(
        args.output_dir / "particle_trajectories.csv",
        tracks,
        fps=args.fps,
        pixel_size_um=args.pixel_size_um,
        frequency_schedule=frequency_schedule,
        rolling_window=args.rolling_window,
    )
    write_summary_csv(args.output_dir / "particle_summary.csv", metrics)
    write_region_counts_csv(args.output_dir / "region_counts.csv", region_counts)
    write_preview(
        args.output_dir / "particle_tracks_preview.png",
        frames[-1],
        tracks,
        roi=roi,
        outlet_x_threshold=args.outlet_x_threshold,
        outlet_center_y=args.outlet_center_y,
        outlet_center_half_height=args.outlet_center_half_height
        if args.outlet_center_y is not None
        else None,
        outlet_inner_half_height=args.outlet_inner_half_height
        if args.outlet_center_y is not None
        else None,
    )

    first_crossover = min(
        (metric for metric in metrics if metric.crossover_frame is not None),
        key=lambda metric: metric.crossover_frame,
        default=None,
    )
    metadata = {
        "input": str(args.input),
        "detector": args.detector,
        "frames": len(frames),
        "fps": args.fps,
        "pixel_size_um": args.pixel_size_um,
        "roi": {
            "x_min": roi_x_min,
            "x_max": roi_x_max,
            "y_min": roi_y_min,
            "y_max": roi_y_max,
        },
        "outlet_classifier": {
            "x_threshold": args.outlet_x_threshold,
            "center_y": args.outlet_center_y,
            "center_half_height": args.outlet_center_half_height,
            "inner_half_height": args.outlet_inner_half_height,
        },
        "total_detected_tracks": len(all_tracks),
        "retained_tracks": len(tracks),
        "highest_track_id": max((track.track_id for track in all_tracks), default=0),
        "first_crossover_track_id": None if first_crossover is None else first_crossover.track_id,
        "first_crossover_frequency_khz": (
            None if first_crossover is None else first_crossover.crossover_frequency_khz
        ),
        "region_counts": region_counts,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Frames processed: {len(frames)}")
    print(f"Total particle IDs assigned: {metadata['highest_track_id']}")
    print(f"Tracks retained after filtering: {len(tracks)}")
    if any(count > 0 for count in region_counts.values()):
        print("Final-region counts:")
        for region in region_order:
            if region_counts.get(region, 0):
                print(f"  {region}: {region_counts[region]}")
    if first_crossover is None:
        print("Crossover: not detected with the current threshold settings.")
    else:
        frequency_label = (
            f"{first_crossover.crossover_frequency_khz:.3f} kHz"
            if first_crossover.crossover_frequency_khz is not None
            else "frequency unavailable"
        )
        print(
            "Crossover: track "
            f"#{first_crossover.track_id:03d} at frame {first_crossover.crossover_frame} "
            f"({first_crossover.crossover_time_s:.3f} s, "
            f"{frequency_label})."
        )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
