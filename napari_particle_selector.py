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
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLineEdit,
    QTextEdit,
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
        self.radio_sorting = QRadioButton(
            "Outlet Sorting Analysis\n"
            "Constriction sorter · Measure fractions in each outlet · Calculate Σ*"
        )
        group = QButtonGroup(self)
        group.addButton(self.radio_tracking)
        group.addButton(self.radio_trapping)
        group.addButton(self.radio_sorting)
        outer.addWidget(self.radio_tracking)
        outer.addWidget(self.radio_trapping)
        outer.addWidget(self.radio_sorting)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def get_mode(self):
        if self.radio_tracking.isChecked():
            return "tracking"
        elif self.radio_sorting.isChecked():
            return "sorting"
        else:
            return "trapping"


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


def load_cnn_detector():
    """Load the trained CNN model if it exists."""
    import torch
    import torch.nn as nn
    model_path = Path("/Users/freddycoronel/Documents/Playground/data/napari_outputs/particle_detector.pt")
    if not model_path.exists():
        return None

    class ParticleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 2),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    model = ParticleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def detect_with_cnn(frame, model, patch_size=32, stride=8, threshold=0.7):
    """Slide CNN across frame and find particle locations."""
    import torch
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
    H, W = gray.shape

    # Mask black borders and post interiors
    border_mask = np.ones((H, W), dtype=np.uint8)
    col_means = gray.mean(axis=0)
    border_mask[:, col_means < 20] = 0
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    post_interior = cv2.erode(dark_mask, kernel)
    search_mask = border_mask & (~post_interior // 255).astype(np.uint8)

    # Only scan bright regions to save time
    bright_regions = gray > 130

    particle_centroids = []
    half = patch_size // 2
    batch_coords, batch_patches = [], []

    for y in range(half, H - half, stride):
        for x in range(half, W - half, stride):
            if not search_mask[y, x]:
                continue
            if not bright_regions[y-2:y+2, x-2:x+2].any():
                continue
            patch = gray[y-half:y+half, x-half:x+half]
            if patch.shape == (patch_size, patch_size):
                batch_coords.append((y, x))
                batch_patches.append(patch)

    if not batch_patches:
        return []

    # Run CNN in batches
    patches_tensor = torch.tensor(
        np.array(batch_patches), dtype=torch.float32
    ).unsqueeze(1) / 255.0

    with torch.no_grad():
        batch_size = 64
        all_probs = []
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.tolist())

    # Find peaks above threshold
    score_map = np.zeros((H, W), dtype=float)
    for (y, x), prob in zip(batch_coords, all_probs):
        if prob > score_map[y, x]:
            score_map[y, x] = prob

    # Non-maximum suppression to get single points
    from scipy.ndimage import maximum_filter
    local_max = (score_map == maximum_filter(score_map, size=patch_size)) & (score_map > threshold)
    ys, xs = np.where(local_max)
    for y, x in zip(ys, xs):
        particle_centroids.append((float(y), float(x)))

    return particle_centroids


def run_trap_counting(*, input_path, fps, pixel_size_um):
    frames = load_video_frames(input_path)
    if frames.size == 0:
        raise ValueError(f"No frames loaded from {input_path}")

    # Use CNN detector if available, otherwise fall back to brightness threshold
    cnn_model = load_cnn_detector()
    if cnn_model is not None:
        print("  Using trained AI detector (CNN)...")
    else:
        print("  Using brightness threshold detector (no trained model found)...")

    print("  Detecting particles across all frames...")
    counts_per_frame = []
    all_centroids = []
    centroids_per_frame = []

    for frame_idx, frame in enumerate(frames):
        # Always run brightness detector
        threshold_centroids = detect_bright_particles(frame)
        if cnn_model is not None:
            # Also run CNN and merge results
            cnn_centroids = detect_with_cnn(frame, cnn_model)
            # Merge: add CNN detections that aren't already covered by threshold
            merged = list(threshold_centroids)
            for cy, cx in cnn_centroids:
                too_close = any(
                    np.sqrt((cy - ty)**2 + (cx - tx)**2) < 15
                    for ty, tx in threshold_centroids
                )
                if not too_close:
                    merged.append((cy, cx))
            centroids = merged
        else:
            centroids = threshold_centroids
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


class AIChatWidget(QWidget):
    def __init__(self, get_context_fn):
        super().__init__()
        self.get_context_fn = get_context_fn
        outer = QVBoxLayout(self)

        title = QLabel("AI Assistant")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer.addWidget(title)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; padding: 8px; "
            "border-radius: 4px; font-size: 11px;"
        )
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setPlainText("Ask me anything about your experiment.")
        outer.addWidget(self.chat_display)

        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask a question...")
        self.input_box.returnPressed.connect(self.send_message)
        input_row.addWidget(self.input_box)

        send_btn = QPushButton("Ask")
        send_btn.clicked.connect(self.send_message)
        input_row.addWidget(send_btn)
        outer.addLayout(input_row)

        self.history = []

    def send_message(self):
        question = self.input_box.text().strip()
        if not question:
            return
        self.input_box.clear()
        self.chat_display.setText("Thinking...")

        context = self.get_context_fn()
        system_prompt = f"""You are an expert AI assistant for DEP (dielectrophoresis) microfluidic experiments.
You have access to the current experiment data:
{context}

Answer questions about the experiment, explain what the data means scientifically, 
guide the user on which particles to mark, and help interpret results.
Keep answers concise and scientific. If asked to summarize, write 2-3 sentences 
suitable for a methods/results section of a paper."""

        import subprocess, json
        self.history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": system_prompt}] + self.history

        try:
            import urllib.request
            payload = json.dumps({
                "model": "mistral",
                "messages": messages,
                "stream": False
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_text = json.loads(resp.read())["message"]["content"]
        except Exception as e:
            response_text = f"Error: {e}. Make sure Ollama is running (run: ollama serve)"

        self.history.append({"role": "assistant", "content": response_text})

        # Show last 3 exchanges
        display = ""
        for msg in self.history[-6:]:
            role = "You" if msg["role"] == "user" else "AI"
            display += f"[{role}]: {msg['content']}\n\n"
        self.chat_display.setPlainText(display.strip())
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def clear_history(self):
        self.history = []
        self.chat_display.setPlainText("Ask me anything about your experiment.")


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

    def retrain_detector(self):
        import subprocess, threading
        patches_dir = self.output_dir / "training_data" / "particles"
        n_patches = len(list(patches_dir.glob("*.png"))) if patches_dir.exists() else 0
        if n_patches < 20:
            QMessageBox.warning(self, "Not enough data",
                f"Only {n_patches} training patches found. Mark at least 20 particles first.")
            return
        self.retrain_status.setText(f"Training on {n_patches} patches... please wait")

        def run_training():
            script = Path("/Users/freddycoronel/Documents/Playground/train_particle_detector.py")
            result = subprocess.run(
                ["python", str(script)],
                capture_output=True, text=True
            )
            # Find best accuracy from output
            acc = "unknown"
            for line in result.stdout.splitlines():
                if "Best validation accuracy" in line:
                    acc = line.split(":")[-1].strip()
            self.retrain_status.setText(f"Training complete! Accuracy: {acc}. Restart app to use new model.")

        threading.Thread(target=run_training, daemon=True).start()

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


class OutletSortingWidget(QWidget):
    def __init__(self, *, viewer, frames, input_path, output_dir, fps, pixel_size_um):
        super().__init__()
        self.viewer        = viewer
        self.frames        = frames
        self.input_path    = input_path
        self.output_dir    = output_dir
        self.fps           = fps
        self.pixel_size_um = pixel_size_um
        self.roi_centers   = {}
        self.intensities   = {}
        self.outlet_names  = ["S1_top", "S2_top", "C", "S2_bottom", "S1_bottom"]

        outer = QVBoxLayout(self)
        title = QLabel("Mode: Outlet Sorting")
        title.setStyleSheet("font-weight: bold;")
        outer.addWidget(title)

        import re
        fname = input_path.stem
        v_match = re.search(r'd(\d+)V', fname)
        f_match = re.search(r'(\d+[Kk]?[Hh]z)', fname)
        self.voltage   = v_match.group(1) + "V" if v_match else "unknown"
        self.frequency = f_match.group(1)         if f_match else "unknown"

        form = QFormLayout()
        self.voltage_label = QLabel(self.voltage)
        self.freq_label    = QLabel(self.frequency)
        self.status_label  = QLabel("Mark 5 outlets then click Analyze")
        self.sigma_label   = QLabel("-")
        self.s1_label      = QLabel("-")
        self.s2_label      = QLabel("-")
        self.c_label       = QLabel("-")
        form.addRow("Voltage",     self.voltage_label)
        form.addRow("Frequency",   self.freq_label)
        form.addRow("Status",      self.status_label)
        form.addRow("Σ*",          self.sigma_label)
        form.addRow("S1 fraction", self.s1_label)
        form.addRow("S2 fraction", self.s2_label)
        form.addRow("C fraction",  self.c_label)
        outer.addLayout(form)

        info = QLabel(
            "To mark outlets:\n"
            "1. Click roi_clicks in layer list\n"
            "2. Press 2 (Add Points mode)\n"
            "3. Click in this order:\n"
            "   S1_top, S2_top, C, S2_bottom, S1_bottom\n"
            "4. Press 3 to exit\n"
            "5. Click Analyze Outlets"
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: gray;")
        outer.addWidget(info)

        self.figure, (self.ax_avg, self.ax_bar) = plt.subplots(1, 2, figsize=(6, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        outer.addWidget(self.canvas)

        buttons = QHBoxLayout()
        analyze_btn = QPushButton("Analyze Outlets")
        analyze_btn.clicked.connect(self.analyze)
        save_btn = QPushButton("Save Results")
        save_btn.clicked.connect(self.save_results)
        buttons.addWidget(analyze_btn)
        buttons.addWidget(save_btn)
        outer.addLayout(buttons)

        avg = np.mean(self.frames, axis=0).astype(np.uint8)
        self.gray_avg = cv2.cvtColor(avg, cv2.COLOR_RGB2GRAY) if avg.ndim == 3 else avg
        self.ax_avg.imshow(self.gray_avg, cmap="gray")
        self.ax_avg.set_title("Time-averaged")
        self.ax_avg.axis("off")
        self.ax_bar.text(0.5, 0.5, "Click outlets\nthen Analyze",
                         ha="center", va="center", transform=self.ax_bar.transAxes)
        self.ax_bar.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_roi_click(self, layer):
        n = len(layer.data)
        if n == 0:
            return
        if n <= len(self.outlet_names):
            name = self.outlet_names[n - 1]
            pt   = layer.data[-1]
            y, x = float(pt[1]), float(pt[2])
            self.roi_centers[name] = (int(y), int(x))
            next_str = f"Next: {self.outlet_names[n]}" if n < 5 else "All done — click Analyze"
            self.status_label.setText(f"Marked {n}/5: {name}. {next_str}")

    def analyze(self):
        if len(self.roi_centers) < 5:
            QMessageBox.warning(self, "Not ready",
                f"Only {len(self.roi_centers)}/5 outlets marked.")
            return

        roi_radius = 10
        H, W = self.gray_avg.shape
        raw = {}

        self.ax_avg.clear()
        self.ax_avg.imshow(self.gray_avg, cmap="gray")
        self.ax_avg.set_title("Time-averaged with ROIs")
        self.ax_avg.axis("off")

        for name in self.outlet_names:
            cy, cx = self.roi_centers[name]
            y0, y1 = max(0, cy-roi_radius), min(H, cy+roi_radius)
            x0, x1 = max(0, cx-roi_radius), min(W, cx+roi_radius)
            raw[name] = float(self.gray_avg[y0:y1, x0:x1].mean())
            circle = plt.Circle((cx, cy), roi_radius,
                                 color="yellow", fill=False, linewidth=1.5)
            self.ax_avg.add_patch(circle)
            self.ax_avg.text(cx+roi_radius+2, cy,
                             f"{name}\n{raw[name]:.0f}",
                             color="yellow", fontsize=7, va="center")

        bg   = min(raw.values())
        corr = {k: max(0.0, v - bg) for k, v in raw.items()}
        self.intensities = corr

        I_S1  = (corr["S1_top"] + corr["S1_bottom"]) / 2
        I_S2  = (corr["S2_top"] + corr["S2_bottom"]) / 2
        I_C   = corr["C"]
        total = I_S1*2 + I_S2*2 + I_C if (I_S1*2 + I_S2*2 + I_C) > 0 else 1

        frac_S1 = I_S1*2 / total
        frac_S2 = I_S2*2 / total
        frac_C  = I_C   / total
        sigma   = (I_S1 - I_C) / (I_S1 + I_C) if (I_S1 + I_C) > 0 else 0

        direction = "S1/S2 (+DEP)" if sigma > 0.05 else "C (-DEP)" if sigma < -0.05 else "No sorting"
        self.sigma_label.setText(f"{sigma:.3f}  ({direction})")
        self.s1_label.setText(f"{frac_S1*100:.1f}%")
        self.s2_label.setText(f"{frac_S2*100:.1f}%")
        self.c_label.setText(f"{frac_C*100:.1f}%")

        self.ax_bar.clear()
        self.ax_bar.bar([0], [frac_S1], color="#e63946", label="S1 outer")
        self.ax_bar.bar([0], [frac_S2], bottom=[frac_S1], color="#f4a261", label="S2 inner")
        self.ax_bar.bar([0], [frac_C],  bottom=[frac_S1+frac_S2], color="#4a9eff", label="C center")
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_ylabel("Fraction")
        self.ax_bar.set_xticks([0])
        self.ax_bar.set_xticklabels([f"{self.voltage}\n{self.frequency}"], fontsize=8)
        self.ax_bar.set_title(f"Σ* = {sigma:.3f}", fontsize=9)
        self.ax_bar.legend(fontsize=7)
        self.ax_bar.set_axis_on()
        self.figure.tight_layout()
        self.canvas.draw()

    def save_results(self):
        if not self.intensities:
            QMessageBox.information(self, "Nothing to save", "Run Analyze first.")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / f"{self.input_path.stem}_sorting"
        rows = [["outlet","bg_corrected_intensity"]]
        for name in self.outlet_names:
            rows.append([name, f"{self.intensities[name]:.2f}"])
        import csv
        with open(str(prefix)+"_results.csv","w",newline="") as f:
            csv.writer(f).writerows(rows)
        self.figure.savefig(str(prefix)+"_plot.png", dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Saved to:\n{self.output_dir}")


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

    # Ask user whether they want a video file or PNG folder
    from qtpy.QtWidgets import QFileDialog, QMessageBox
    choice = QMessageBox()
    choice.setWindowTitle("Select input type")
    choice.setText("What are you loading?")
    btn_file   = choice.addButton("Video file (.avi/.gif/.mp4)", QMessageBox.AcceptRole)
    btn_folder = choice.addButton("PNG image folder", QMessageBox.AcceptRole)
    choice.addButton("Cancel", QMessageBox.RejectRole)
    choice.exec_()

    clicked = choice.clickedButton()
    if clicked == btn_file:
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select video file",
            str(Path.home() / "Downloads"),
            "Supported Files (*.gif *.mp4 *.avi *.mov *.tif *.tiff);;All Files (*)",
        )
        if file_path:
            args.input = Path(file_path)
            print(f"File selected: {args.input}")
    elif clicked == btn_folder:
        folder_path = QFileDialog.getExistingDirectory(
            None, "Select folder of PNG images",
            str(Path.home() / "Downloads"),
        )
        if folder_path:
            args.input = Path(folder_path)
            print(f"Folder selected: {args.input}")
    else:
        print("Cancelled.")
        return

    if mode == "project":
        # Load saved project file
        from qtpy.QtWidgets import QFileDialog
        proj_path, _ = QFileDialog.getOpenFileName(
            None, "Open Project File",
            str(Path.home() / "Downloads"),
            "DEP Project Files (*.dep);;All Files (*)",
        )
        if not proj_path:
            print("No project selected.")
            return
        project = json.loads(Path(proj_path).read_text())
        input_path    = Path(project["input_path"])
        fps           = project["fps"]
        pixel_size_um = project["pixel_size_um"]
        counts_per_frame = project["counts_per_frame"]
        centroids_per_frame = [
            [(y, x) for y, x in frame]
            for frame in project["centroids_per_frame"]
        ]
        all_centroids = project.get("all_centroids", [])
        manual_seeds  = [(y, x) for y, x in project.get("manual_seeds", [])]
        manual_trajs  = [
            [(t[0], t[1], t[2], t[3]) for t in traj]
            for traj in project.get("manual_trajectories", [])
        ]

        print(f"Loading project: {proj_path}")
        frames = load_video_frames(input_path)

        viewer = napari.Viewer(title=f"Trap Counting — {input_path.name} [Project]")
        viewer.add_image(frames, name="frames", rgb=True)

        if all_centroids:
            pt_data = np.asarray(all_centroids, dtype=float)
            auto_points_layer = viewer.add_points(
                pt_data, name="trapped_particles",
                size=6, face_color="red", border_color="white", opacity=0.7,
                out_of_slice_display=True,
            )
        else:
            auto_points_layer = None

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
            output_dir=args.output_dir,
            input_path=input_path,
            fps=fps, pixel_size_um=pixel_size_um,
        )
        widget.manual_layer  = manual_layer
        widget.manual_seeds  = manual_seeds
        widget.manual_trajectories = manual_trajs
        widget.manual_label.setText(str(len(manual_seeds)))

        # Restore manual trajectories as path shapes
        if manual_trajs:
            for traj in manual_trajs:
                path_pts = np.array([[t[1], t[2]] for t in traj])
                if "manual_paths" not in [l.name for l in viewer.layers]:
                    viewer.add_shapes(
                        [path_pts], shape_type="path",
                        edge_color="lime", edge_width=2,
                        name="manual_paths", opacity=0.8,
                    )
                else:
                    paths_layer = viewer.layers["manual_paths"]
                    existing = list(paths_layer.data)
                    existing.append(path_pts)
                    paths_layer.data = existing

        viewer.window.add_dock_widget(widget, area="right", name="Trap Counting")

        # Add AI chatbox
        def get_context():
            lines = [
                f"Video: {args.input.name}",
                f"Frames analyzed: {len(counts_per_frame)}",
                f"Mean particles/frame: {np.mean(counts_per_frame):.1f}",
                f"Peak count: {max(counts_per_frame)} at t={np.argmax(counts_per_frame)/args.fps:.1f}s",
                f"Total detections: {sum(counts_per_frame)}",
            ]
            if hasattr(widget, 'flow_label'):
                lines.append(f"Mean flow speed: {widget.flow_label.text()}")
                lines.append(f"Flow half-1 (vx): {widget.flow_h1_label.text()}")
                lines.append(f"Flow half-2 (vx): {widget.flow_h2_label.text()}")
            if widget.selected_label.text() != "None":
                lines.append(f"Selected particle: {widget.selected_label.text()}")
            if widget.manual_seeds:
                lines.append(f"Manually marked particles: {len(widget.manual_seeds)}")
            return "\n".join(lines)

        chat_widget = AIChatWidget(get_context_fn=get_context)
        viewer.window.add_dock_widget(chat_widget, area="right", name="AI Assistant")

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
        def _on_manual_add_proj(_event=None):
            if len(manual_layer.data) == 0:
                return
            pt = manual_layer.data[-1]
            y, x = float(pt[1]), float(pt[2])
            traj = follow_particle(y, x, centroids_per_frame, max_dist=20)
            path_pts = np.array([[t[1], t[2]] for t in traj])
            if "manual_paths" not in [l.name for l in viewer.layers]:
                viewer.add_shapes(
                    [path_pts], shape_type="path",
                    edge_color="lime", edge_width=2,
                    name="manual_paths", opacity=0.8,
                )
            else:
                paths_layer = viewer.layers["manual_paths"]
                existing = list(paths_layer.data)
                existing.append(path_pts)
                paths_layer.data = existing
            widget.on_manual_add(y, x)

        napari.run()
        return

    if mode == "sorting":
        print("Loading video for outlet sorting...")
        frames = load_video_frames(args.input)
        viewer = napari.Viewer(title=f"Outlet Sorting — {args.input.name}")
        avg = np.mean(frames, axis=0).astype(np.uint8)
        viewer.add_image(frames, name="frames", rgb=True)
        viewer.add_image(avg, name="time_average", rgb=True, opacity=0.5)
        roi_layer = viewer.add_points(
            np.zeros((0, 3), dtype=float),
            name="roi_clicks",
            size=10, face_color="yellow", border_color="red", opacity=0.9,
        )
        widget = OutletSortingWidget(
            viewer=viewer, frames=frames,
            input_path=args.input, output_dir=args.output_dir,
            fps=args.fps, pixel_size_um=args.pixel_size_um,
        )
        viewer.window.add_dock_widget(widget, area="right", name="Outlet Sorting")

        @roi_layer.events.data.connect
        def _on_roi(_event=None):
            widget.on_roi_click(roi_layer)

        napari.run()
        return

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
                out_of_slice_display=True,
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

        # Add AI chatbox
        def get_context():
            lines = [
                f"Video: {args.input.name}",
                f"Frames analyzed: {len(counts_per_frame)}",
                f"Mean particles/frame: {np.mean(counts_per_frame):.1f}",
                f"Peak count: {max(counts_per_frame)} at t={np.argmax(counts_per_frame)/args.fps:.1f}s",
                f"Total detections: {sum(counts_per_frame)}",
            ]
            if hasattr(widget, 'flow_label'):
                lines.append(f"Mean flow speed: {widget.flow_label.text()}")
                lines.append(f"Flow half-1 (vx): {widget.flow_h1_label.text()}")
                lines.append(f"Flow half-2 (vx): {widget.flow_h2_label.text()}")
            if widget.selected_label.text() != "None":
                lines.append(f"Selected particle: {widget.selected_label.text()}")
            if widget.manual_seeds:
                lines.append(f"Manually marked particles: {len(widget.manual_seeds)}")
            return "\n".join(lines)

        chat_widget = AIChatWidget(get_context_fn=get_context)
        viewer.window.add_dock_widget(chat_widget, area="right", name="AI Assistant")

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

        # Keep track of how many seeds added
        _n_seeds = [0]

        @manual_layer.events.data.connect
        def _on_manual_add(_event=None):
            if len(manual_layer.data) == 0:
                return
            # Only fire on a genuinely new point
            if len(manual_layer.data) <= _n_seeds[0]:
                return
            _n_seeds[0] = len(manual_layer.data)
            pt = manual_layer.data[-1]
            y, x = float(pt[1]), float(pt[2])
            print(f"  Marked particle at ({int(x)}, {int(y)})")
            # Add this point at every frame so it persists when scrubbing
            all_frame_pts = np.array([[f, y, x] for f in range(len(frames))])
            manual_layer.events.data.disconnect(_on_manual_add)
            existing = manual_layer.data[:-1] if len(manual_layer.data) > 1 else np.zeros((0,3))
            manual_layer.data = np.vstack([existing, all_frame_pts]) if len(existing) > 0 else all_frame_pts
            _n_seeds[0] = len(manual_layer.data)
            manual_layer.events.data.connect(_on_manual_add)
            manual_layer.opacity = 0.9
            manual_layer.refresh()
            print(f"  Green dot added at all frames, total points: {len(manual_layer.data)}")
            # Save training patch at clicked location
            widget._save_training_patch(y, x)
            widget.on_manual_add(y, x)

    napari.run()


if __name__ == "__main__":
    main()
