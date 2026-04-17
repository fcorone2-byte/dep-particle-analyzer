#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from analyze_particle_video import load_frames
from dep_video_tracking.detectors import ThresholdBlobDetector
from dep_video_tracking.physics import (
    FrequencySchedule,
    instantaneous_speeds_um_s,
    rolling_mean,
    summarize_track,
)
from dep_video_tracking.tracking import NearestNeighborTracker, Track


class ModeDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Analysis Mode")
        self.setMinimumWidth(380)
        outer = QVBoxLayout(self)
        title = QLabel("What type of experiment are you analyzing?")
        title.setWordWrap(True)
        outer.addWidget(title)
        self.radio_tracking = QRadioButton(
            "Particle Tracking\n"
            "Constriction sorter · DEP crossover frequency · Velocity analysis"
        )
        self.radio_tracking.setChecked(True)
        self.radio_trapping = QRadioButton(
            "Trap Counting\n"
            "Post array · iDEP trapping · Count + select + manually mark particles"
        )
        group = QButtonGroup(self)
        group.addButton(self.radio_tracking)
        group.addButton(self.radio_trapping)
        outer.addWidget(self.radio_tracking)
        outer.addWidget(self.radio_trapping)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def get_mode(self):
        return "tracking" if self.radio_tracking.isChecked() else "trapping"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("/Users/freddycoronel/Downloads/Updated_Linear_multiple_releases (1).gif"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--pixel-size-um", type=float, default=0.32)
    parser.add_argument("--start-frequency-khz", type=float, default=50.0)
    parser.add_argument("--stop-frequency-khz", type=float, default=250.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/Users/freddycoronel/Documents/Playground/data/napari_outputs"))
    return parser.parse_args()


def load_video_frames(input_path):
    return np.asarray(load_frames(input_path))


def run_tracking(*, input_path, fps, pixel_size_um, start_frequency_khz, stop_frequency_khz):
    frames = load_video_frames(input_path)
    if frames.size == 0:
        raise ValueError(f"No frames loaded from {input_path}")
    detector = ThresholdBlobDetector(
        color_spread_min=8, color_max_channel=200,
        min_component_area=2, max_component_area=800,
    )
    tracker = NearestNeighborTracker(max_distance_px=18.0, max_missed_frames=4)
    for frame_idx, frame in enumerate(frames):
        tracker.update(frame_idx=frame_idx, detections=detector.detect(frame))
    tracks = [t for t in tracker.finalize() if len(t.points) >= 3]
    if not tracks:
        raise RuntimeError("No tracks found. Try Trap Counting mode.")
    schedule = FrequencySchedule(
        start_khz=start_frequency_khz, stop_khz=stop_frequency_khz,
        num_frames=len(frames),
    )
    summary_kwargs = dict(
        pixel_size_um=pixel_size_um, fps=fps,
        velocity_threshold_um_s=2.0, stall_frames=3, rolling_window=3,
        frequency_schedule=schedule, outlet_x_threshold=None,
        outlet_center_y=None, outlet_center_half_height=35.0,
        outlet_inner_half_height=68.0,
    )
    track_info = {}
    for track in tracks:
        speeds = instantaneous_speeds_um_s(track, pixel_size_um=pixel_size_um, fps=fps)
        track_info[track.track_id] = {
            "track": track,
            "metrics": asdict(summarize_track(track, **summary_kwargs)),
            "speeds": speeds,
            "smoothed": rolling_mean(speeds, 3),
        }
    return frames, tracks, track_info, schedule


def tracks_to_napari_arrays(tracks):
    track_rows, point_rows, point_track_ids = [], [], []
    for track in tracks:
        for point in track.points:
            track_rows.append([track.track_id, point.frame_idx, point.y_px, point.x_px])
            point_rows.append([point.frame_idx, point.y_px, point.x_px])
            point_track_ids.append(track.track_id)
    track_data = np.asarray(track_rows, dtype=float) if track_rows else np.zeros((0, 4), dtype=float)
    point_data = np.asarray(point_rows, dtype=float) if point_rows else np.zeros((0, 3), dtype=float)
    return track_data, point_data, {"track_id": np.asarray(point_track_ids, dtype=int)}


def detect_bright_particles(frame, percentile_threshold=98.5, min_area=3, max_area=400):
    """
    Detect trapped DNA particles as very bright spots.
    Uses adaptive per-frame percentile threshold so particles are found
    consistently even as overall brightness changes across frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
    H, W = gray.shape

    # Mask out black borders
    border_mask = np.ones((H, W), dtype=np.uint8)
    col_means = gray.mean(axis=0)
    border_mask[:, col_means < 20] = 0

    # Mask out post interiors (large dark regions)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    post_interior = cv2.erode(dark_mask, kernel)
    search_mask = border_mask & (~post_interior // 255).astype(np.uint8)

    # Adaptive threshold: top percentile_threshold% of valid pixels
    valid_pixels = gray[search_mask == 1]
    if valid_pixels.size == 0:
        return []
    threshold = np.percentile(valid_pixels, percentile_threshold)

    # Also enforce a minimum absolute brightness to avoid noise
    threshold = max(threshold, 130)

    bright = ((gray > threshold) & (search_mask == 1)).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
    particle_centroids = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx_p, cy_p = centroids[label]
            particle_centroids.append((float(cy_p), float(cx_p)))

    return particle_centroids


def run_trap_counting(*, input_path, fps, pixel_size_um):
    frames = load_video_frames(input_path)
    if frames.size == 0:
        raise ValueError(f"No frames loaded from {input_path}")

    print("  Detecting particles across all frames...")
    counts_per_frame = []
    all_centroids = []
    centroids_per_frame = []

    for frame_idx, frame in enumerate(frames):
        centroids = detect_bright_particles(frame)
        counts_per_frame.append(len(centroids))
        centroids_per_frame.append(centroids)
        for (y, x) in centroids:
            all_centroids.append([frame_idx, y, x])

        if frame_idx % 10 == 0:
            print(f"    Frame {frame_idx}/{len(frames)} — {len(centroids)} particles")

    print(f"  Done. Mean particles/frame: {np.mean(counts_per_frame):.1f}")
    return frames, counts_per_frame, all_centroids, centroids_per_frame


def calculate_flow_velocity(centroids_per_frame, fps, pixel_size_um,
                            min_displacement=5, max_displacement=80):
    """
    Calculate flow velocity of moving particles by tracking displacement
    between consecutive frames. Particles moving more than min_displacement
    pixels are considered flowing (not trapped).
    Returns per-frame mean velocity (signed, in um/s) and per-particle velocities.
    """
    frame_velocities = []
    all_vx = []
    all_vy = []

    for f_idx in range(1, len(centroids_per_frame)):
        prev = centroids_per_frame[f_idx - 1]
        curr = centroids_per_frame[f_idx]
        if not prev or not curr:
            frame_velocities.append((0.0, 0.0))
            continue

        prev_pts = np.array(prev)
        curr_pts = np.array(curr)

        frame_vx, frame_vy = [], []
        used = set()
        for i, (py, px) in enumerate(prev_pts):
            dists = np.sqrt((curr_pts[:,1] - px)**2 + (curr_pts[:,0] - py)**2)
            nearest = np.argmin(dists)
            if nearest in used:
                continue
            d = dists[nearest]
            if min_displacement <= d <= max_displacement:
                dy = (curr_pts[nearest][0] - py) * pixel_size_um * fps
                dx = (curr_pts[nearest][1] - px) * pixel_size_um * fps
                frame_vx.append(dx)
                frame_vy.append(dy)
                all_vx.append(dx)
                all_vy.append(dy)
                used.add(nearest)

        mean_vx = np.mean(frame_vx) if frame_vx else 0.0
        mean_vy = np.mean(frame_vy) if frame_vy else 0.0
        frame_velocities.append((mean_vx, mean_vy))

    return frame_velocities, all_vx, all_vy


def follow_particle(seed_y, seed_x, centroids_per_frame, max_dist=20):
    trajectory = []
    cy, cx = seed_y, seed_x
    for frame_idx, frame_centroids in enumerate(centroids_per_frame):
        if not frame_centroids:
            trajectory.append((frame_idx, cy, cx, False))
            continue
        pts = np.array(frame_centroids)
        dists = np.sqrt((pts[:, 1] - cx)**2 + (pts[:, 0] - cy)**2)
        nearest_idx = np.argmin(dists)
        if dists[nearest_idx] < max_dist:
            cy, cx = pts[nearest_idx][0], pts[nearest_idx][1]
            trajectory.append((frame_idx, cy, cx, True))
        else:
            trajectory.append((frame_idx, cy, cx, False))
    return trajectory


class TrackingWidget(QWidget):
    def __init__(self, *, viewer, track_info, output_dir, input_path,
                 fps, pixel_size_um, start_frequency_khz, stop_frequency_khz):
        super().__init__()
        self.viewer = viewer
        self.track_info = track_info
        self.output_dir = output_dir
        self.input_path = input_path
        self.fps = fps
        self.pixel_size_um = pixel_size_um
        self.start_frequency_khz = start_frequency_khz
        self.stop_frequency_khz = stop_frequency_khz
        self.selected_track_id = None
        outer = QVBoxLayout(self)
        mode_label = QLabel("Mode: Particle Tracking")
        mode_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(mode_label)
        form = QFormLayout()
        self.track_id_label   = QLabel("-")
        self.num_points_label = QLabel("-")
        self.mean_speed_label = QLabel("-")
        self.max_speed_label  = QLabel("-")
        self.net_disp_label   = QLabel("-")
        self.crossover_label  = QLabel("-")
        form.addRow("Track ID",              self.track_id_label)
        form.addRow("Points",                self.num_points_label)
        form.addRow("Mean speed (um/s)",     self.mean_speed_label)
        form.addRow("Max speed (um/s)",      self.max_speed_label)
        form.addRow("Net displacement (um)", self.net_disp_label)
        form.addRow("Crossover",             self.crossover_label)
        outer.addLayout(form)
        self.figure, (self.ax_track, self.ax_speed) = plt.subplots(2, 1, figsize=(4.8, 6.8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        outer.addWidget(self.canvas)
        buttons = QHBoxLayout()
        for label, slot in [("Save Selected Track", self.save_selected_track),
                             ("Open Another File",   self.open_new_file)]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        outer.addLayout(buttons)
        self._draw_empty()

    def _draw_empty(self):
        for ax, msg in [(self.ax_track, "Click a point in the viewer."),
                        (self.ax_speed,  "Velocity plot will appear here.")]:
            ax.clear()
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_selected_track(self, track_id):
        if track_id not in self.track_info:
            return
        self.selected_track_id = track_id
        info     = self.track_info[track_id]
        track    = info["track"]
        metrics  = info["metrics"]
        speeds   = info["speeds"]
        smoothed = info["smoothed"]
        points   = track.points
        self.track_id_label.setText(str(track_id))
        self.num_points_label.setText(str(metrics["num_points"]))
        self.mean_speed_label.setText(f"{metrics['mean_speed_um_s']:.3f}")
        self.max_speed_label.setText(f"{metrics['max_speed_um_s']:.3f}")
        self.net_disp_label.setText(f"{metrics['net_displacement_um']:.3f}")
        xo_time = metrics["crossover_time_s"]
        self.crossover_label.setText(
            "Not detected" if xo_time is None
            else f"{xo_time:.3f} s | {metrics['crossover_frequency_khz']:.3f} kHz"
        )
        self.ax_track.clear()
        self.ax_track.plot([pt.x_px for pt in points], [pt.y_px for pt in points],
                           color="red", linewidth=2)
        self.ax_track.scatter([pt.x_px for pt in points], [pt.y_px for pt in points],
                              color="gold", s=14)
        self.ax_track.set_title(f"Selected track #{track_id}")
        self.ax_track.invert_yaxis()
        self.ax_track.set_xlabel("x (px)")
        self.ax_track.set_ylabel("y (px)")
        self.ax_speed.clear()
        if speeds:
            times = np.asarray([pt.frame_idx / self.fps for pt in points[1:]], dtype=float)
            self.ax_speed.plot(times, speeds,   label="Instantaneous speed", alpha=0.45)
            self.ax_speed.plot(times, smoothed, label="Smoothed speed",      linewidth=2)
            self.ax_speed.axhline(2.0, color="red", linestyle="--", label="XO threshold")
            if xo_time is not None:
                self.ax_speed.axvline(xo_time, color="purple", linestyle=":",
                                      label="Detected crossover")
            self.ax_speed.set_xlabel("Time (s)")
            self.ax_speed.set_ylabel("Speed (um/s)")
            self.ax_speed.legend(loc="best")
            self.ax_speed.set_title("Velocity vs time")
        else:
            self.ax_speed.text(0.5, 0.5, "Track too short.",
                               ha="center", va="center", transform=self.ax_speed.transAxes)
            self.ax_speed.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_selected_track(self):
        if self.selected_track_id is None:
            QMessageBox.information(self, "No selection", "Click a particle first.")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        info     = self.track_info[self.selected_track_id]
        track    = info["track"]
        metrics  = info["metrics"]
        speeds   = info["speeds"]
        smoothed = info["smoothed"]
        points   = track.points
        ps       = self.pixel_size_um
        prefix   = self.output_dir / f"{self.input_path.stem}_track_{self.selected_track_id}"
        np.savetxt(
            prefix.with_name(prefix.name + "_trajectory.csv"),
            np.asarray([[track.track_id, pt.frame_idx, pt.frame_idx / self.fps,
                         pt.x_px, pt.y_px, pt.x_px * ps, pt.y_px * ps, pt.confidence]
                        for pt in points], dtype=float),
            delimiter=",", header="track_id,frame,time_s,x_px,y_px,x_um,y_um,confidence",
            comments="",
        )
        np.savetxt(
            prefix.with_name(prefix.name + "_speed.csv"),
            np.asarray([[points[i+1].frame_idx, points[i+1].frame_idx / self.fps,
                         spd, smoothed[i]] for i, spd in enumerate(speeds)], dtype=float),
            delimiter=",", header="frame,time_s,instantaneous_speed_um_s,smoothed_speed_um_s",
            comments="",
        )
        prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(metrics, indent=2))
        prefix.with_name(prefix.name + "_metadata.json").write_text(json.dumps({
            "input": str(self.input_path), "selected_track_id": self.selected_track_id,
            "fps": self.fps, "pixel_size_um": ps,
            "start_frequency_khz": self.start_frequency_khz,
            "stop_frequency_khz": self.stop_frequency_khz,
        }, indent=2))
        QMessageBox.information(self, "Saved", f"Saved to:\n{self.output_dir}")

    def open_new_file(self):
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose GIF/video", str(self.input_path.parent),
            "Supported Files (*.gif *.mp4 *.avi *.mov *.tif *.tiff);;All Files (*)",
        )
        if selected:
            QMessageBox.information(self, "Restart required",
                f'Relaunch with:\npython3 napari_particle_selector.py --input "{selected}"')


class TrapCountingWidget(QWidget):
    def __init__(self, *, viewer, frames, counts_per_frame,
                 all_centroids, centroids_per_frame,
                 output_dir, input_path, fps, pixel_size_um):
        super().__init__()
        self.viewer              = viewer
        self.frames              = frames
        self.counts_per_frame    = counts_per_frame
        self.centroids_per_frame = centroids_per_frame
        self.output_dir          = output_dir
        self.input_path          = input_path
        self.fps                 = fps
        self.pixel_size_um       = pixel_size_um
        self.selected_layer      = None
        self.manual_layer        = None
        self.manual_seeds        = []
        self.manual_trajectories = []

        outer = QVBoxLayout(self)
        mode_label = QLabel("Mode: Trap Counting")
        mode_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(mode_label)

        form = QFormLayout()
        self.frames_label    = QLabel(str(len(counts_per_frame)))
        self.total_label     = QLabel(str(sum(counts_per_frame)))
        self.mean_label      = QLabel(f"{np.mean(counts_per_frame):.1f}")
        self.peak_label      = QLabel(f"{max(counts_per_frame)}")
        self.selected_label  = QLabel("None")
        self.manual_label    = QLabel("0")
        form.addRow("Frames analyzed",    self.frames_label)
        form.addRow("Total detections",   self.total_label)
        form.addRow("Mean count / frame", self.mean_label)
        form.addRow("Peak count",         self.peak_label)
        form.addRow("Selected particle",  self.selected_label)
        form.addRow("Manually marked",    self.manual_label)
        self.flow_label    = QLabel("-")
        self.flow_h1_label = QLabel("-")
        self.flow_h2_label = QLabel("-")
        form.addRow("Mean flow speed",    self.flow_label)
        form.addRow("Flow half-1 (vx)",   self.flow_h1_label)
        form.addRow("Flow half-2 (vx)",   self.flow_h2_label)
        outer.addLayout(form)

        info_label = QLabel(
            "Click a red dot to see its trap history.\n\n"
            "To mark a missed particle:\n"
            "1. Go to frame 0 (slider to left)\n"
            "2. Click manual_marks in layer list\n"
            "3. Press 2 (Add Points mode)\n"
            "4. Click on the particle\n"
            "5. Press 3 to exit"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px; color: gray;")
        outer.addWidget(info_label)

        self.figure, (self.ax_count, self.ax_trap) = plt.subplots(2, 1, figsize=(4.8, 5.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        outer.addWidget(self.canvas)

        buttons = QHBoxLayout()
        for label, slot in [("Save Report", self.save_report),
                             ("Open Another File", self.open_new_file)]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        outer.addLayout(buttons)
        self._draw_summary()

    def _draw_summary(self):
        counts = self.counts_per_frame
        times  = np.arange(len(counts)) / self.fps

        # Calculate flow velocities
        frame_vels, all_vx, all_vy = calculate_flow_velocity(
            self.centroids_per_frame, self.fps, self.pixel_size_um)

        # Update flow metrics labels
        if all_vx:
            mean_speed = np.mean(np.sqrt(np.array(all_vx)**2 + np.array(all_vy)**2))
            self.flow_label.setText(f"{mean_speed:.2f} μm/s")
            # Split into two half-periods
            mid = len(frame_vels) // 2
            h1_vx = [v[0] for v in frame_vels[:mid] if v[0] != 0]
            h2_vx = [v[0] for v in frame_vels[mid:] if v[0] != 0]
            h1_mean = np.mean(h1_vx) if h1_vx else 0
            h2_mean = np.mean(h2_vx) if h2_vx else 0
            self.flow_h1_label.setText(f"{h1_mean:+.2f} μm/s")
            self.flow_h2_label.setText(f"{h2_mean:+.2f} μm/s")
        else:
            self.flow_label.setText("N/A")
            self.flow_h1_label.setText("N/A")
            self.flow_h2_label.setText("N/A")

        self.ax_count.clear()
        self.ax_count.plot(times, counts, color="#4a9eff", linewidth=1.5)
        self.ax_count.fill_between(times, counts, alpha=0.2, color="#4a9eff")
        self.ax_count.set_xlabel("Time (s)")
        self.ax_count.set_ylabel("Particles detected")
        self.ax_count.set_title("Trap count over time")

        # Flow velocity plot
        vel_times = np.arange(1, len(frame_vels)+1) / self.fps
        vx_vals = [v[0] for v in frame_vels]
        self.ax_trap.clear()
        self.ax_trap.plot(vel_times, vx_vals, color="#34d399", linewidth=1.2, alpha=0.8)
        self.ax_trap.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        self.ax_trap.axvline(len(frame_vels)/(2*self.fps), color="#fbbf24",
                             linewidth=1, linestyle="--", alpha=0.6, label="half-period")
        self.ax_trap.fill_between(vel_times, vx_vals, 0, alpha=0.15, color="#34d399")
        self.ax_trap.set_xlabel("Time (s)")
        self.ax_trap.set_ylabel("Flow vx (μm/s)")
        self.ax_trap.set_title("Flow velocity over time")
        self.ax_trap.legend(fontsize=8)
        self.ax_trap.set_axis_on()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_selected_particle(self, y, x, label="auto"):
        traj     = follow_particle(y, x, self.centroids_per_frame)
        times    = np.array([t[0] / self.fps for t in traj])
        detected = np.array([t[3] for t in traj], dtype=float)
        trap_pct = detected.mean() * 100

        # Calculate per-frame velocity for this particle
        xs = np.array([t[2] for t in traj]) * self.pixel_size_um
        ys = np.array([t[1] for t in traj]) * self.pixel_size_um
        dx = np.diff(xs) * self.fps
        dy = np.diff(ys) * self.fps
        speeds = np.sqrt(dx**2 + dy**2)
        mean_spd = np.mean(speeds[detected[1:] == 1]) if detected[1:].sum() > 0 else 0.0

        self.selected_label.setText(
            f"({int(x)}, {int(y)}) — present {trap_pct:.0f}%  |  mean v = {mean_spd:.2f} μm/s")

        # Store for group averaging
        if not hasattr(self, "_selected_speeds"):
            self._selected_speeds = []
        self._selected_speeds.append(speeds)
        if len(self._selected_speeds) > 1:
            group_mean = np.mean([s.mean() for s in self._selected_speeds])
            self.selected_label.setText(
                self.selected_label.text() +
                f"  |  group mean = {group_mean:.2f} μm/s ({len(self._selected_speeds)} particles)")

        # Top plot: trap count summary + trajectory overlay
        counts    = self.counts_per_frame
        all_times = np.arange(len(counts)) / self.fps
        self.ax_count.clear()
        self.ax_count.plot(all_times, counts, color="#4a9eff", linewidth=1,
                           alpha=0.4, label="All particles")
        self.ax_count.fill_between(all_times, counts, alpha=0.1, color="#4a9eff")
        self.ax_count.set_xlabel("Time (s)")
        self.ax_count.set_ylabel("Particles detected")
        self.ax_count.set_title("Trap count over time")

        # Bottom plot: trajectory path + velocity
        self.ax_trap.clear()
        # Draw trajectory as x position over time
        traj_times = np.array([t[0] / self.fps for t in traj])
        traj_x     = np.array([t[2] for t in traj]) * self.pixel_size_um
        traj_y     = np.array([t[1] for t in traj]) * self.pixel_size_um

        ax2 = self.ax_trap
        ax2.plot(traj_x, traj_y, color="#f4a261", linewidth=1.5, alpha=0.8)
        ax2.scatter([traj_x[0]], [traj_y[0]], color="lime",   s=30, zorder=5, label="start")
        ax2.scatter([traj_x[-1]], [traj_y[-1]], color="red",  s=30, zorder=5, label="end")
        ax2.invert_yaxis()
        ax2.set_xlabel("x (μm)")
        ax2.set_ylabel("y (μm)")
        ax2.set_title(f"Trajectory — {trap_pct:.0f}% trapped  |  v̄ = {mean_spd:.2f} μm/s")
        ax2.legend(fontsize=7, loc="upper right")
        ax2.set_axis_on()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def reset_selection(self):
        """Clear group selection history."""
        if hasattr(self, "_selected_speeds"):
            self._selected_speeds = []

    def on_auto_click(self, y, x):
        self.show_selected_particle(y, x, label="auto")

    def on_manual_add(self, y, x):
        self.manual_seeds.append((y, x))
        self.manual_trajectories.append(follow_particle(y, x, self.centroids_per_frame))
        self.manual_label.setText(str(len(self.manual_seeds)))
        self.show_selected_particle(y, x, label="manual")
        self._save_training_patch(y, x)

    def _save_training_patch(self, y, x, patch_size=32):
        """Save image patch around manually marked particle for future AI training."""
        try:
            import os
            train_dir = self.output_dir / "training_data" / "particles"
            train_dir.mkdir(parents=True, exist_ok=True)
            # Get current frame
            current_frame_idx = int(self.viewer.dims.point[0])
            frame = self.frames[current_frame_idx]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
            H, W = gray.shape
            # Extract patch around the marked particle
            y0 = max(0, int(y) - patch_size)
            y1 = min(H, int(y) + patch_size)
            x0 = max(0, int(x) - patch_size)
            x1 = min(W, int(x) + patch_size)
            patch = gray[y0:y1, x0:x1]
            # Save with unique filename
            n_existing = len(list(train_dir.glob("*.png")))
            patch_path = train_dir / f"particle_{n_existing+1:04d}_f{current_frame_idx}_y{int(y)}_x{int(x)}.png"
            cv2.imwrite(str(patch_path), patch)
            print(f"  Training patch saved: {patch_path.name}")
        except Exception as e:
            print(f"  Could not save training patch: {e}")

    def save_report(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / f"{self.input_path.stem}_trap_count"
        times = np.arange(len(self.counts_per_frame)) / self.fps
        np.savetxt(
            prefix.with_name(prefix.name + "_counts.csv"),
            np.column_stack([times, self.counts_per_frame]),
            delimiter=",", header="time_s,particle_count", comments="",
        )
        for idx, (traj, seed) in enumerate(zip(self.manual_trajectories, self.manual_seeds)):
            np.savetxt(
                prefix.with_name(prefix.name + f"_manual_{idx+1}_trajectory.csv"),
                np.asarray([[t[0], t[0]/self.fps, t[2], t[1],
                             t[2]*self.pixel_size_um, t[1]*self.pixel_size_um, int(t[3])]
                            for t in traj], dtype=float),
                delimiter=",",
                header="frame,time_s,x_px,y_px,x_um,y_um,detected",
                comments="",
            )
        prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps({
            "input": str(self.input_path), "fps": self.fps,
            "pixel_size_um": self.pixel_size_um,
            "frames_analyzed": len(self.counts_per_frame),
            "total_detections": int(sum(self.counts_per_frame)),
            "mean_count_per_frame": float(np.mean(self.counts_per_frame)),
            "peak_count": int(max(self.counts_per_frame)),
            "manually_marked_particles": len(self.manual_seeds),
        }, indent=2))
        self.figure.savefig(
            prefix.with_name(prefix.name + "_plots.png"), dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Report saved to:\n{self.output_dir}")

    def open_new_file(self):
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose video", str(self.input_path.parent),
            "Supported Files (*.gif *.mp4 *.avi *.mov *.tif *.tiff);;All Files (*)",
        )
        if selected:
            QMessageBox.information(self, "Restart required",
                f'Relaunch with:\npython3 napari_particle_selector.py --input "{selected}"')


def main():
    args = parse_args()
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)

    dialog = ModeDialog()
    if dialog.exec_() != QDialog.Accepted:
        print("Cancelled.")
        return
    mode = dialog.get_mode()
    print(f"Mode selected: {mode}")

    # Ask user to pick a file
    from qtpy.QtWidgets import QFileDialog
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select video file to analyze",
        str(Path.home() / "Downloads"),
        "Supported Files (*.gif *.mp4 *.avi *.mov *.tif *.tiff);;All Files (*)",
    )
    if file_path:
        args.input = Path(file_path)
        print(f"File selected: {args.input}")
    else:
        print("No file selected — using default.")

    if mode == "tracking":
        frames, tracks, track_info, _schedule = run_tracking(
            input_path=args.input,
            fps=args.fps,
            pixel_size_um=args.pixel_size_um,
            start_frequency_khz=args.start_frequency_khz,
            stop_frequency_khz=args.stop_frequency_khz,
        )
        viewer = napari.Viewer(title=f"Particle Tracking — {args.input.name}")
        viewer.add_image(frames, name="frames", rgb=True)
        track_data, point_data, point_properties = tracks_to_napari_arrays(tracks)
        viewer.add_tracks(
            track_data, name="tracks",
            properties={"track_id": track_data[:, 0].astype(int)} if len(track_data) else None,
            tail_length=30, head_length=0, blending="translucent",
        )
        points_layer = viewer.add_points(
            point_data, name="track_points", properties=point_properties,
            size=8, face_color="cyan", border_color="white", opacity=0.8,
        )
        widget = TrackingWidget(
            viewer=viewer, track_info=track_info, output_dir=args.output_dir,
            input_path=args.input, fps=args.fps, pixel_size_um=args.pixel_size_um,
            start_frequency_khz=args.start_frequency_khz,
            stop_frequency_khz=args.stop_frequency_khz,
        )
        viewer.window.add_dock_widget(widget, area="right", name="Particle Tracking")

        @points_layer.events.highlight.connect
        def _handle_selection(_event=None):
            selected = list(points_layer.selected_data)
            if not selected:
                return
            track_id = int(points_layer.properties["track_id"][min(selected)])
            widget.set_selected_track(track_id)

    else:
        print("Running trap counting — this may take a moment...")
        frames, counts_per_frame, all_centroids, centroids_per_frame = run_trap_counting(
            input_path=args.input,
            fps=args.fps,
            pixel_size_um=args.pixel_size_um,
        )

        viewer = napari.Viewer(title=f"Trap Counting — {args.input.name}")
        viewer.add_image(frames, name="frames", rgb=True)

        auto_points_layer = None
        if all_centroids:
            pt_data = np.asarray(all_centroids, dtype=float)
            auto_points_layer = viewer.add_points(
                pt_data, name="trapped_particles",
                size=6, face_color="red", border_color="white", opacity=0.7,
            )

        manual_layer = viewer.add_points(
            np.zeros((0, 3), dtype=float),
            name="manual_marks",
            size=12, face_color="lime", border_color="white", opacity=0.9,
            out_of_slice_display=True,
        )

        widget = TrapCountingWidget(
            viewer=viewer, frames=frames,
            counts_per_frame=counts_per_frame,
            all_centroids=all_centroids,
            centroids_per_frame=centroids_per_frame,
            output_dir=args.output_dir, input_path=args.input,
            fps=args.fps, pixel_size_um=args.pixel_size_um,
        )
        widget.manual_layer = manual_layer
        viewer.window.add_dock_widget(widget, area="right", name="Trap Counting")

        if auto_points_layer is not None:
            def _on_select_change(event):
                selected = list(auto_points_layer.selected_data)
                if not selected:
                    return
                idx = min(selected)
                pt = auto_points_layer.data[idx]
                y, x = float(pt[1]), float(pt[2])
                widget.on_auto_click(y, x)
            auto_points_layer.events.connect(_on_select_change)

        @manual_layer.events.data.connect
        def _on_manual_add(_event=None):
            if len(manual_layer.data) == 0:
                return
            pt = manual_layer.data[-1]
            y, x = float(pt[1]), float(pt[2])
            # Follow particle across all frames
            traj = follow_particle(y, x, centroids_per_frame, max_dist=20)
            # Build points for all frames so dot persists when scrubbing
            all_pts = np.array([[t[0], t[1], t[2]] for t in traj])
            # Disconnect to avoid recursion, replace data, reconnect
            manual_layer.events.data.disconnect(_on_manual_add)
            prev = manual_layer.data[:-1] if len(manual_layer.data) > 1 else np.zeros((0,3))
            manual_layer.data = np.vstack([prev, all_pts]) if len(prev) > 0 else all_pts
            manual_layer.events.data.connect(_on_manual_add)
            widget.on_manual_add(y, x)

    napari.run()


if __name__ == "__main__":
    main()
