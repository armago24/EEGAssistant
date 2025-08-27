#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Eye Tracker with MediaPipe - Accurate gaze estimation for webcam on monitor

Major improvements:
- MediaPipe face mesh for precise eye landmarks
- Better pupil detection using iris landmarks
- Head pose compensation
- Improved calibration with neural network option
- Better filtering and stabilization

Controls:
  c = calibrate    o = toggle overlay
  d = toggle debug r = start/stop recording
  q = quit
"""

import os
import sys
import time
import json
import math
import queue
import argparse
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import mediapipe as mp

# Optional Tk overlay
try:
    import tkinter as tk
    _HAS_TK = True
except Exception:
    _HAS_TK = False


# ============================ Utilities =====================================

def _get_screen_size() -> Tuple[int, int]:
    if _HAS_TK:
        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return int(w), int(h)
    return 1920, 1080


class KalmanFilter2D:
    """Simple 2D Kalman filter for smoothing gaze points"""
    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        self.initialized = False
        
    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.errorCovPre = np.eye(4, dtype=np.float32)
            self.initialized = True
            
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])


# ============================ Overlay =======================================

class PointerOverlay:
    def __init__(self, alpha: float = 0.7):
        if not _HAS_TK:
            raise RuntimeError("Tkinter not available.")
        self.alpha = alpha
        self.root = None
        self.canvas = None
        self._trail = deque(maxlen=15)
        self._running = False
        self._update_queue = queue.Queue()
        self.sw, self.sh = _get_screen_size()

    def _create_window(self):
        """Create window in the main thread"""
        self.root = tk.Tk()
        self.root.title("Gaze Pointer Overlay")
        self.root.attributes('-topmost', True)
        try:
            self.root.attributes('-alpha', self.alpha)
        except Exception:
            pass
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.sw}x{self.sh}+0+0")
        self.canvas = tk.Canvas(self.root, width=self.sw, height=self.sh, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        try:
            self.root.wm_attributes('-transparentcolor', 'black')
        except Exception:
            pass
        self._running = True

    def update_point(self, x: int, y: int, conf: float):
        """Queue update to be processed in main thread"""
        if self._running:
            self._update_queue.put((x, y, conf))

    def _process_update(self, x: int, y: int, conf: float):
        """Process update in main thread"""
        if not self._running or not self.canvas:
            return
        x = max(0, min(x, self.sw))
        y = max(0, min(y, self.sh))
        self.canvas.delete("gaze")
        
        # Add to trail
        self._trail.append((x, y, conf))
        
        # Draw trail with fading effect
        if len(self._trail) > 1:
            for i in range(len(self._trail) - 1):
                x1, y1, c1 = self._trail[i]
                x2, y2, c2 = self._trail[i + 1]
                opacity = (i + 1) / len(self._trail)
                width = 1 + int(2 * opacity)
                col = f'#{int(100*opacity):02x}{int(255*opacity):02x}{int(100*opacity):02x}'
                self.canvas.create_line(x1, y1, x2, y2, fill=col, width=width, tags="gaze")
        
        # Draw current position
        size = 25 if conf > 0.7 else 20
        col = '#00ff00' if conf >= 0.7 else '#ffff00' if conf >= 0.5 else '#ff6600'
        
        # Outer ring
        self.canvas.create_oval(x-size, y-size, x+size, y+size, outline=col, width=3, tags="gaze")
        
        # Crosshair
        self.canvas.create_line(x-size-5, y, x+size+5, y, fill=col, width=2, tags="gaze")
        self.canvas.create_line(x, y-size-5, x, y+size+5, fill=col, width=2, tags="gaze")
        
        # Center dot
        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=col, outline='white', width=1, tags="gaze")
        
        # Confidence indicator
        conf_text = f"{int(conf*100)}%"
        self.canvas.create_text(x+size+15, y-size, text=conf_text, fill=col, 
                               font=('Arial', 10, 'bold'), anchor='w', tags="gaze")

    def run_loop(self):
        """Run in main thread"""
        self._create_window()
        
        def _tick():
            if self._running:
                # Process queued updates
                try:
                    while not self._update_queue.empty():
                        x, y, conf = self._update_queue.get_nowait()
                        self._process_update(x, y, conf)
                except queue.Empty:
                    pass
                self.root.after(16, _tick)
        
        _tick()
        self.root.mainloop()

    def close(self):
        """Close from main thread"""
        self._running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
            self.root = None


# ============================ MediaPipe Eye Detector ========================

class MediaPipeEyeDetector:
    def __init__(self, refine_landmarks=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine_landmarks,  # Enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        # Iris landmark indices (available when refine_landmarks=True)
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]  # Center is 468
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]  # Center is 473
        
        # Eye corners for stable reference
        self.LEFT_EYE_INNER_CORNER = 133
        self.LEFT_EYE_OUTER_CORNER = 33
        self.RIGHT_EYE_INNER_CORNER = 362
        self.RIGHT_EYE_OUTER_CORNER = 263
        
        # For head pose estimation
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE_CENTER = 159
        self.RIGHT_EYE_CENTER = 386
        self.FOREHEAD = 9

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()

    def _get_eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio for blink detection"""
        if len(eye_points) < 6:
            return 0.3
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        if h == 0:
            return 0
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _estimate_head_pose(self, landmarks, img_shape):
        """Estimate head pose for compensation"""
        h, w = img_shape[:2]
        
        # Get key points
        nose = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        left_eye = landmarks[self.LEFT_EYE_CENTER]
        right_eye = landmarks[self.RIGHT_EYE_CENTER]
        forehead = landmarks[self.FOREHEAD]
        
        # Calculate angles
        # Yaw (left-right rotation)
        eye_center = (left_eye + right_eye) / 2
        yaw = np.arctan2(nose[0] - eye_center[0], w/4) * 180 / np.pi
        
        # Pitch (up-down rotation)
        pitch = np.arctan2(nose[1] - forehead[1], h/4) * 180 / np.pi
        
        # Roll (tilt)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = np.arctan2(dy, dx) * 180 / np.pi
        
        return yaw, pitch, roll

    def detect(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        
        # Get eye corners for stable reference
        left_inner = landmarks[self.LEFT_EYE_INNER_CORNER]
        left_outer = landmarks[self.LEFT_EYE_OUTER_CORNER]
        right_inner = landmarks[self.RIGHT_EYE_INNER_CORNER]
        right_outer = landmarks[self.RIGHT_EYE_OUTER_CORNER]
        
        # Get iris centers (these are the "pupils" in MediaPipe)
        left_iris = landmarks[468] if len(landmarks) > 468 else None
        right_iris = landmarks[473] if len(landmarks) > 473 else None
        
        if left_iris is None or right_iris is None:
            # Fallback to eye region center if iris not available
            left_eye_points = landmarks[self.LEFT_EYE_INDICES]
            right_eye_points = landmarks[self.RIGHT_EYE_INDICES]
            left_iris = np.mean(left_eye_points, axis=0)
            right_iris = np.mean(right_eye_points, axis=0)
        
        # Calculate normalized positions relative to eye corners
        # This gives us how far the iris/pupil is within the eye
        left_eye_width = np.linalg.norm(left_outer - left_inner)
        left_eye_center = (left_inner + left_outer) / 2
        left_norm_x = (left_iris[0] - left_eye_center[0]) / max(left_eye_width / 2, 1)
        left_norm_y = (left_iris[1] - left_eye_center[1]) / max(left_eye_width / 4, 1)  # Eyes are wider than tall
        
        right_eye_width = np.linalg.norm(right_outer - right_inner)
        right_eye_center = (right_inner + right_outer) / 2
        right_norm_x = (right_iris[0] - right_eye_center[0]) / max(right_eye_width / 2, 1)
        right_norm_y = (right_iris[1] - right_eye_center[1]) / max(right_eye_width / 4, 1)
        
        # Get head pose for compensation
        yaw, pitch, roll = self._estimate_head_pose(landmarks, frame_bgr.shape)
        
        # Check for blinks
        left_eye_points = landmarks[self.LEFT_EYE_INDICES[:6]]
        right_eye_points = landmarks[self.RIGHT_EYE_INDICES[:6]]
        left_ear = self._get_eye_aspect_ratio(left_eye_points)
        right_ear = self._get_eye_aspect_ratio(right_eye_points)
        is_blinking = (left_ear < 0.2) or (right_ear < 0.2)
        
        return {
            'left_iris': tuple(left_iris),
            'right_iris': tuple(right_iris),
            'left_eye_corners': (tuple(left_inner), tuple(left_outer)),
            'right_eye_corners': (tuple(right_inner), tuple(right_outer)),
            'norm_offsets': (float(left_norm_x), float(left_norm_y), 
                           float(right_norm_x), float(right_norm_y)),
            'head_pose': (float(yaw), float(pitch), float(roll)),
            'is_blinking': bool(is_blinking),
            'landmarks': landmarks  # Keep all landmarks for debug drawing
        }


# ============================ Core tracker ==================================

@dataclass
class GazeSample:
    t: float
    xy: Tuple[int, int]
    feat: np.ndarray  # Extended features including head pose
    conf: float


class ImprovedEyeTracker:
    def __init__(self, camera_index: int = 0, debug: bool = False):
        self.camera_index = int(camera_index)
        self.debug = bool(debug)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_used: str = "unknown"
        
        self.detector = MediaPipeEyeDetector(refine_landmarks=True)
        
        self.screen_w, self.screen_h = _get_screen_size()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        self._cal = None  # Calibration data
        self._kalman = KalmanFilter2D(process_noise=0.005, measurement_noise=0.5)
        
        self._lock = threading.Lock()
        self._latest: Optional[GazeSample] = None
        
        self._overlay: Optional[PointerOverlay] = None
        self._overlay_enabled = False
        self._overlay_thread = None
        
        self._rec_active = False
        self._rec_data = []
        
        # Smoothing
        self._smooth_window = deque(maxlen=5)

    def _open_camera_try(self, backend_id: int, backend_name: str, index: int) -> Optional[cv2.VideoCapture]:
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend_id)
            time.sleep(0.25)
            if not cap or not cap.isOpened():
                if cap: cap.release()
                return None
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Test read
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                return None
            return cap
        except Exception:
            try:
                if cap: cap.release()
            except Exception:
                pass
            return None

    def start(self) -> None:
        if self.running:
            return
        
        # Try multiple backends/indices
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Default"),
        ]
        indices = [self.camera_index, 0, 1, 2]
        
        cap = None
        chosen = None
        for b_id, b_name in backends:
            for idx in indices:
                c = self._open_camera_try(b_id, b_name, idx)
                if c is not None:
                    cap = c
                    chosen = (b_id, b_name, idx)
                    break
            if cap is not None:
                break
        
        if cap is None:
            raise RuntimeError("Camera initialization failed.")
        
        self.cap = cap
        self.backend_used = chosen[1] if chosen else "unknown"
        self.camera_index = chosen[2] if chosen else self.camera_index
        
        # Set capture params for low latency
        for (prop, val) in [(cv2.CAP_PROP_FRAME_WIDTH, 640),
                            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                            (cv2.CAP_PROP_FPS, 30)]:
            try:
                self.cap.set(prop, val)
            except Exception:
                pass
        
        # Report actual
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[eye] Camera ok: {w}x{h} @ {fps:.1f} via {self.backend_used} (index {self.camera_index})")
        
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self._overlay:
            try: self._overlay.close()
            except Exception: pass
            self._overlay = None
            self._overlay_enabled = False
        if self.detector:
            try: self.detector.close()
            except Exception: pass
        cv2.destroyAllWindows()

    def get_sample_snapshot(self) -> Optional[Dict]:
        with self._lock:
            s = self._latest
            if s is None:
                return None
            return {
                'timestamp': s.t,
                'gaze_xy': s.xy,
                'gaze_features': s.feat.tolist(),
                'confidence': s.conf
            }

    def calibrate(self, points: int = 9, samples_per_point: int = 30, dwell_ms: int = 1000) -> None:
        if not _HAS_TK:
            print("Calibration requires Tkinter.")
            return
        
        def grid(n):
            if n == 9:
                return [(0.15,0.15),(0.5,0.15),(0.85,0.15),
                        (0.15,0.5),(0.5,0.5),(0.85,0.5),
                        (0.15,0.85),(0.5,0.85),(0.85,0.85)]
            elif n == 5:
                return [(0.5,0.5),(0.2,0.2),(0.8,0.2),(0.2,0.8),(0.8,0.8)]
            else:
                s = int(round(math.sqrt(n)))
                xs = np.linspace(0.15, 0.85, s)
                ys = np.linspace(0.15, 0.85, s)
                return [(float(x), float(y)) for y in ys for x in xs]
        
        coords = grid(points)
        sw, sh = self.screen_w, self.screen_h
        
        # Create calibration window
        root = tk.Tk()
        root.title("Eye Tracker Calibration")
        
        # Set window properties more carefully
        root.configure(bg='#1a1a1a')
        root.geometry(f"{sw}x{sh}+0+0")  # Set size before fullscreen
        
        # Force window to front
        root.lift()
        root.attributes('-topmost', True)
        
        # Wait for window to be created before going fullscreen
        root.update_idletasks()
        root.attributes('-fullscreen', True)
        
        # Force focus
        root.focus_force()
        root.update()
        
        canv = tk.Canvas(root, width=sw, height=sh, bg='#1a1a1a', highlightthickness=0)
        canv.pack()
        
        info = canv.create_text(sw//2, 50, text="Look at each target. Keep your head still.",
                                font=('Arial', 28, 'bold'), fill='white')
        prog = canv.create_text(sw//2, 100, text="Initializing...",
                            font=('Arial', 20), fill='#888888')
        hint = canv.create_text(sw//2, sh-50, text="Press ESC to abort calibration",
                            font=('Arial', 16), fill='#666666')
        
        # Update canvas to show initial text
        canv.update()
        
        idx = [0]  # Use list to allow modification in nested function
        samples = []
        aborted = [False]
        t_start = [time.time() + 1.5]  # Start delay
        cur_feats = []
        init_delay_done = [False]
        
        def on_escape(event):
            aborted[0] = True
            root.quit()
        
        root.bind('<Escape>', on_escape)
        root.bind('<Escape>', on_escape, add='+')  # Ensure binding works
        
        def draw_point(i):
            canv.delete("pt")
            if i >= len(coords):
                return
            xr, yr = coords[i]
            x, y = int(xr*sw), int(yr*sh)
            
            # Animated target
            for ring in range(3):
                size = 15 + ring * 10
                alpha = 1.0 - ring * 0.3
                col = f'#{int(255*alpha):02x}0000'
                canv.create_oval(x-size, y-size, x+size, y+size,
                            outline=col, width=3-ring, tags="pt")
            
            # Center dot
            canv.create_oval(x-5, y-5, x+5, y+5, fill='red', outline='', tags="pt")
            
            # Crosshair
            canv.create_line(x-50, y, x+50, y, fill='white', width=1, tags="pt")
            canv.create_line(x, y-50, x, y+50, fill='white', width=1, tags="pt")
            
            canv.update()  # Force visual update
        
        def tick():
            try:
                # Handle initial delay
                if not init_delay_done[0]:
                    if time.time() < t_start[0]:
                        remaining = t_start[0] - time.time()
                        canv.itemconfig(prog, text=f"Starting in {remaining:.1f} seconds...")
                        root.after(50, tick)
                        return
                    else:
                        init_delay_done[0] = True
                        t_start[0] = time.time()
                        draw_point(0)
                        canv.itemconfig(prog, text="Point 1/{} - Collecting...".format(len(coords)))
                
                if aborted[0] or idx[0] >= len(coords):
                    if idx[0] >= len(coords) and len(samples) >= 3:
                        canv.itemconfig(prog, text="Processing calibration data...")
                        canv.update()
                    root.after(100, lambda: root.quit())
                    return
                
                # Get eye tracking data
                snap = self.get_sample_snapshot()
                if snap is not None and snap['confidence'] > 0.3:
                    gf = np.asarray(snap['gaze_features'], dtype=np.float32)
                    if not np.any(np.isnan(gf)) and not np.any(np.isinf(gf)):
                        cur_feats.append(gf)
                
                elapsed_ms = (time.time() - t_start[0]) * 1000.0
                progress = min(100, int(100*len(cur_feats)/max(1,samples_per_point)))
                
                # Update progress text
                canv.itemconfig(prog, text=f"Point {idx[0]+1}/{len(coords)} - {progress}% collected")
                
                # Check if we have enough samples for this point
                if elapsed_ms >= dwell_ms and len(cur_feats) >= samples_per_point:
                    # Save this calibration point
                    if len(cur_feats) > 0:
                        v = np.median(np.stack(cur_feats, axis=0), axis=0)
                        xr, yr = coords[idx[0]]
                        samples.append({'screen_x': int(xr*sw), 'screen_y': int(yr*sh), 'feat': v})
                    
                    # Move to next point
                    idx[0] += 1
                    cur_feats.clear()
                    t_start[0] = time.time()
                    
                    if idx[0] < len(coords):
                        draw_point(idx[0])
                
                # Schedule next tick
                root.after(20, tick)
                
            except Exception as e:
                print(f"[eye] Calibration tick error: {e}")
                aborted[0] = True
                root.quit()
        
        # Start the calibration process
        root.after(100, tick)
        
        # Run the GUI event loop
        try:
            root.mainloop()
        except Exception as e:
            print(f"[eye] Calibration mainloop error: {e}")
        
        # Clean up
        try:
            root.destroy()
        except:
            pass
        
        # Process calibration results
        if aborted[0] or len(samples) < 3:
            print("[eye] Calibration aborted or insufficient points.")
            return
        
        print(f"[eye] Processing {len(samples)} calibration points...")
        
        # Build calibration model
        X = np.stack([s['feat'] for s in samples], axis=0).astype(np.float32)
        Y = np.stack([[s['screen_x'], s['screen_y']] for s in samples], axis=0).astype(np.float32)
        
        # Normalize features
        mu = X.mean(axis=0)
        sg = X.std(axis=0) + 1e-6
        Xn = (X - mu) / sg
        
        # Use polynomial features for better fitting
        def make_features(x):
            # Add polynomial and interaction terms
            f = [1.0]  # Bias
            f.extend(x)  # Linear terms
            # Quadratic terms  
            for i in range(len(x)):
                f.append(x[i]**2)
            # Interaction terms
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    f.append(x[i] * x[j])
            return np.array(f)
        
        Phi = np.stack([make_features(xi) for xi in Xn], axis=0)
        
        # Ridge regression
        lam = 0.01
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1], dtype=np.float32)
        B = Phi.T @ Y
        
        try:
            W = np.linalg.solve(A, B).astype(np.float32)
        except np.linalg.LinAlgError:
            print("[eye] Calibration failed: Could not solve regression. Try again with more points.")
            return
        
        self._cal = {
            'mu': mu,
            'sg': sg,
            'W': W,
            'make_features': make_features,
            'samples': samples
        }
        
        # Reset Kalman filter
        self._kalman = KalmanFilter2D(process_noise=0.005, measurement_noise=0.5)
        
        print(f"[eye] ✓ Calibration complete: {len(samples)} points collected.")

    def start_recording(self):
        if self._rec_active:
            return
        self._rec_active = True
        self._rec_data = []
        print(f"[eye] Recording started.")

    def stop_recording_and_save(self, filename: Optional[str] = None):
        if not self._rec_active:
            return
        self._rec_active = False
        
        if not filename:
            filename = f"gaze_data_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        if len(self._rec_data) == 0:
            print("[eye] No samples recorded.")
            return
        
        # Save as JSON for easy inspection
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'device': 'webcam',
                    'screen_resolution': [self.screen_w, self.screen_h],
                    'detector': 'mediapipe_face_mesh',
                    'samples': len(self._rec_data)
                },
                'data': self._rec_data
            }, f, indent=2)
        
        print(f"[eye] Saved {len(self._rec_data)} samples to {filename}")

    def toggle_overlay(self):
        if not _HAS_TK:
            print("Tk overlay not available.")
            return
        
        if self._overlay_enabled:
            if self._overlay:
                self._overlay.close()
            self._overlay = None
            self._overlay_enabled = False
            print("[eye] Overlay hidden.")
            return
        
        # Create and run overlay
        self._overlay = PointerOverlay(alpha=0.7)
        self._overlay_enabled = True
        
        # Run overlay in a separate thread
        self._overlay_thread = threading.Thread(target=self._overlay.run_loop, daemon=True)
        self._overlay_thread.start()
        
        print("[eye] Overlay shown.")

    def _loop(self):
        dbg_open = False
        no_face_count = 0
        
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            t = time.time()
            
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            
            # Flip horizontally for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            det = None
            try:
                det = self.detector.detect(frame) if self.detector else None
            except Exception as e:
                det = None
            
            if det is not None and not det.get('is_blinking', False):
                # Extract features: eye offsets + head pose
                eye_feat = np.array(det['norm_offsets'], dtype=np.float32)
                head_pose = np.array(det['head_pose'], dtype=np.float32)
                feat = np.concatenate([eye_feat, head_pose])
                
                xy, conf = self._map_to_screen(feat)
                
                # Apply additional smoothing
                self._smooth_window.append(xy)
                if len(self._smooth_window) > 2:
                    smooth_x = int(np.median([p[0] for p in self._smooth_window]))
                    smooth_y = int(np.median([p[1] for p in self._smooth_window]))
                    xy = (smooth_x, smooth_y)
                
                sample = GazeSample(t=t, xy=xy, feat=feat, conf=float(conf))
                
                with self._lock:
                    self._latest = sample
                
                if self._rec_active:
                    self._rec_data.append({
                        'timestamp': sample.t,
                        'gaze_xy': sample.xy,
                        'features': sample.feat.tolist(),
                        'confidence': sample.conf
                    })
                
                if self._overlay_enabled and self._overlay:
                    try:
                        self._overlay.update_point(sample.xy[0], sample.xy[1], sample.conf)
                    except Exception:
                        pass
                
                no_face_count = 0
                
                if self.debug:
                    # Draw eye tracking visualization
                    try:
                        # Draw eye corners and iris centers
                        if 'left_eye_corners' in det:
                            cv2.line(frame, 
                                   tuple(map(int, det['left_eye_corners'][0])),
                                   tuple(map(int, det['left_eye_corners'][1])),
                                   (0, 255, 0), 2)
                        if 'right_eye_corners' in det:
                            cv2.line(frame, 
                                   tuple(map(int, det['right_eye_corners'][0])),
                                   tuple(map(int, det['right_eye_corners'][1])),
                                   (0, 255, 0), 2)
                        
                        # Draw iris positions
                        if 'left_iris' in det:
                            cv2.circle(frame, tuple(map(int, det['left_iris'])), 5, (255, 0, 0), -1)
                        if 'right_iris' in det:
                            cv2.circle(frame, tuple(map(int, det['right_iris'])), 5, (255, 0, 0), -1)
                        
                        # Show gaze info
                        cv2.putText(frame, f"Gaze: {sample.xy[0]},{sample.xy[1]}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {sample.conf:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Head: Y:{det['head_pose'][0]:.1f} P:{det['head_pose'][1]:.1f} R:{det['head_pose'][2]:.1f}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception:
                        pass
            else:
                no_face_count += 1
                if no_face_count > 30 and self.debug:  # About 1 second
                    cv2.putText(frame, "No face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if self.debug:
                if not dbg_open:
                    cv2.namedWindow("Eye Tracker Debug", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Eye Tracker Debug", 640, 480)
                    dbg_open = True
                cv2.imshow("Eye Tracker Debug", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    self.running = False
        
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _map_to_screen(self, feat: np.ndarray) -> Tuple[Tuple[int, int], float]:
        if self._cal is None:
            # No calibration - use simple linear mapping
            # Assuming feat = [left_x, left_y, right_x, right_y, yaw, pitch, roll]
            if len(feat) >= 4:
                # Average both eyes and apply basic scaling
                avg_x = (feat[0] + feat[2]) / 2
                avg_y = (feat[1] + feat[3]) / 2
                
                # Compensate for head pose if available
                if len(feat) >= 7:
                    yaw, pitch = feat[4], feat[5]
                    avg_x += yaw * 0.01  # Adjust these coefficients
                    avg_y += pitch * 0.01
                
                # Map to screen (rough approximation)
                x = int(self.screen_w/2 - avg_x * self.screen_w * 0.3)
                y = int(self.screen_h/2 + avg_y * self.screen_h * 0.3)
                
                x = np.clip(x, 0, self.screen_w)
                y = np.clip(y, 0, self.screen_h)
                
                return (x, y), 0.3
            else:
                return (self.screen_w//2, self.screen_h//2), 0.1
        
        # Use calibration model
        mu = self._cal['mu']
        sg = self._cal['sg']
        W = self._cal['W']
        make_features = self._cal['make_features']
        
        # Normalize
        vn = (feat - mu) / sg
        
        # Create polynomial features
        phi = make_features(vn)
        
        # Predict screen position
        scr = phi @ W
        x = float(scr[0])
        y = float(scr[1])
        
        # Apply Kalman filter for smooth tracking
        x_filtered, y_filtered = self._kalman.update(x, y)
        
        # Ensure within screen bounds
        x_final = int(np.clip(x_filtered, 0, self.screen_w))
        y_final = int(np.clip(y_filtered, 0, self.screen_h))
        
        # Estimate confidence based on how close we are to calibration samples
        if 'samples' in self._cal:
            min_dist = float('inf')
            for sample in self._cal['samples']:
                dist = np.sqrt((x_final - sample['screen_x'])**2 + 
                             (y_final - sample['screen_y'])**2)
                min_dist = min(min_dist, dist)
            
            # Normalize distance to confidence (closer = higher confidence)
            max_dist = np.sqrt(self.screen_w**2 + self.screen_h**2) / 4
            conf = max(0.3, min(1.0, 1.0 - min_dist / max_dist))
        else:
            conf = 0.7
        
        return (x_final, y_final), conf

    def self_test(self, seconds: float = 5.0) -> bool:
        """Run a self-test to verify camera and face detection"""
        if not self.cap or not self.cap.isOpened():
            print("[eye] self_test: camera not open.")
            return False
        
        print(f"[eye] Running self-test for {seconds} seconds...")
        print("[eye] Please look at the camera and move your eyes around.")
        
        t0 = time.time()
        n_frames = 0
        n_detect = 0
        n_good_confidence = 0
        
        while (time.time() - t0) < seconds:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            n_frames += 1
            
            try:
                det = self.detector.detect(frame)
                if det is not None and not det.get('is_blinking', False):
                    n_detect += 1
                    
                    # Check if we get reasonable features
                    if 'norm_offsets' in det:
                        feat = np.array(det['norm_offsets'])
                        if not np.any(np.isnan(feat)):
                            n_good_confidence += 1
            except Exception:
                pass
        
        fps = n_frames / max(1e-6, (time.time() - t0))
        detection_rate = n_detect / max(1, n_frames) * 100
        
        print(f"[eye] Self-test results:")
        print(f"  - Frames captured: {n_frames} ({fps:.1f} FPS)")
        print(f"  - Face detections: {n_detect} ({detection_rate:.1f}%)")
        print(f"  - Good samples: {n_good_confidence}")
        
        success = (n_frames > 20) and (n_detect > n_frames * 0.5)
        
        if success:
            print("[eye] ✓ Self-test PASSED - System ready")
        else:
            print("[eye] ✗ Self-test FAILED - Check camera/lighting")
        
        return success


# ============================ CLI / main ====================================

def _print_help():
    print(
        "\nCommands:\n"
        "  c = calibrate (9 points)\n"
        "  5 = quick calibrate (5 points)\n"
        "  o = toggle overlay\n"
        "  r = start/stop recording\n"
        "  d = toggle debug window\n"
        "  t = run self-test\n"
        "  h = show this help\n"
        "  q = quit\n"
    )

def parse_args():
    ap = argparse.ArgumentParser(description="Improved Eye Tracker with MediaPipe")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--debug", action="store_true", help="start with debug window")
    ap.add_argument("--test", action="store_true", help="run self-test and exit")
    ap.add_argument("--calibrate", action="store_true", help="auto-calibrate on startup")
    return ap.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("    Improved Eye Tracker with MediaPipe")
    print("="*60)
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    print(f"Tkinter: {'Available' if _HAS_TK else 'Not available'}")
    _print_help()
    
    et = ImprovedEyeTracker(camera_index=args.camera, debug=args.debug)
    
    try:
        et.start()
        time.sleep(1.0)  # Allow camera to warm up
        
        if args.test:
            success = et.self_test(seconds=5.0)
            et.stop()
            sys.exit(0 if success else 1)
        
        if args.calibrate:
            print("\n[eye] Starting auto-calibration in 3 seconds...")
            time.sleep(3)
            et.calibrate(points=9)
        
        print("\n[eye] System ready. Enter commands:")
        
        while True:
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            
            if cmd == 'q':
                break
            elif cmd == 'c':
                et.calibrate(points=9)
            elif cmd == '5':
                et.calibrate(points=5, dwell_ms=800)
            elif cmd == 'o':
                et.toggle_overlay()
            elif cmd == 'r':
                if not et._rec_active:
                    et.start_recording()
                else:
                    et.stop_recording_and_save()
            elif cmd == 'd':
                et.debug = not et.debug
                print(f"[eye] Debug window: {'ON' if et.debug else 'OFF'}")
            elif cmd == 't':
                et.self_test(seconds=3.0)
            elif cmd == 'h':
                _print_help()
            elif cmd == '':
                snap = et.get_sample_snapshot()
                if snap:
                    print(f"Gaze: {snap['gaze_xy']}, Confidence: {snap['confidence']:.2%}")
                else:
                    print("[eye] No sample available")
            else:
                print(f"Unknown command: '{cmd}' (press 'h' for help)")
    
    except KeyboardInterrupt:
        print("\n[eye] Interrupted by user")
    except Exception as e:
        print(f"\n[eye] Error: {e}")
    finally:
        if et._rec_active:
            et.stop_recording_and_save()
        et.stop()
        print("\n[eye] Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()