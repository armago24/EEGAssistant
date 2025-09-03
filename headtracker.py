#!/usr/bin/env python3
"""
Head-Directed Gaze Approximation (Standalone, robust camera open)
- MediaPipe Face Mesh -> head pose via solvePnP (no world landmarks)
- One Euro Filter for low-lag smoothing
- 5-point calibration maps yaw/pitch -> normalized (x,y)
- Robust camera open on Windows: tries DirectShow, MSMF, VFW, with timeouts
- Optional OSC out: /head/ypr, /gaze/norm, /gaze/screen
"""

import argparse
import json
import os
import time
from collections import deque
import struct
import threading

import cv2
import numpy as np

# -------------------- Screen size --------------------
def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080

SCREEN_W, SCREEN_H = get_screen_size()

# -------------------- MediaPipe ----------------------
try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh
except ImportError as e:
    raise SystemExit("mediapipe is required. Install: pip install mediapipe") from e

# MediaPipe landmark indices
FACE_IDX = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_outer": 263,   # subject's left (image right)
    "right_eye_outer": 33,   # subject's right (image left)
    "left_mouth": 291,
    "right_mouth": 61,
}

# ---------------- One Euro Filter --------------------
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        now = time.time() if t is None else t
        if self.t_prev is None:
            self.t_prev = now
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dt = max(now - self.t_prev, 1e-6)
        self.freq = 1.0 / dt
        self.t_prev = now
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * (self.dx_prev if self.dx_prev is not None else dx)
        self.dx_prev = dx_hat
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

# --------------- Minimal OSC sender ------------------
class OSCSender:
    def __init__(self, host='127.0.0.1', port=8053, enabled=False):
        import socket
        self.addr = (host, port)
        self.enabled = enabled
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _pad4(b: bytes) -> bytes:
        return b + (b'\x00' * ((4 - (len(b) % 4)) % 4))

    def _pack(self, address: str, args):
        b = b''
        b += self._pad4(address.encode('ascii'))
        tags = ',' + ''.join('i' if isinstance(a, int) else 'f' for a in args)
        b += self._pad4(tags.encode('ascii'))
        for a in args:
            b += struct.pack('>i', a) if isinstance(a, int) else struct.pack('>f', float(a))
        return b

    def send(self, address: str, *args):
        if not self.enabled:
            return
        try:
            self.sock.sendto(self._pack(address, args), self.addr)
        except Exception:
            pass

# -------- Calibration: yaw/pitch -> (x,y) ------------
class CalibrationModel:
    # 2x3 matrix M so [x,y]^T = M @ [yaw, pitch, 1]^T
    def __init__(self, M: np.ndarray):
        self.M = np.array(M, dtype=np.float64).reshape(2, 3)

    def to_json(self): return {"M": self.M.tolist()}
    @staticmethod
    def from_json(d): return CalibrationModel(np.array(d["M"], dtype=np.float64))
    def map(self, yaw_deg: float, pitch_deg: float):
        x, y = (self.M @ np.array([yaw_deg, pitch_deg, 1.0], dtype=np.float64))
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

def fit_calibration(yaw_list, pitch_list, xy_targets):
    X = np.column_stack([yaw_list, pitch_list, np.ones(len(yaw_list))])
    Y = np.array(xy_targets, dtype=np.float64)
    M_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return CalibrationModel(M_T.T)

# --------------- Head pose helpers -------------------
def rotation_matrix_to_euler_ypr(R: np.ndarray):
    """
    Compute yaw (Y), pitch (X), roll (Z) in degrees from a 3x3 rotation matrix.
    Conventions:
      - yaw:   left/right, +left
      - pitch: up/down,    +up
      - roll:  head tilt,  +clockwise (as seen by the camera)
    Implementation:
      - forward vector f = R[:, 2]
      - yaw   = atan2(fx, fz)
      - pitch = atan2(-fy, sqrt(fx^2 + fz^2))
      - roll  = atan2(R[0,1], R[1,1])   (from the 'up' column)
    """
    # Forward (camera Z in head coords projected to camera frame)
    fx, fy, fz = R[0, 2], R[1, 2], R[2, 2]
    # Guard against invalid matrices
    denom = max(np.sqrt(fx*fx + fz*fz), 1e-8)

    yaw   = np.degrees(np.arctan2(fx, fz))
    pitch = np.degrees(np.arctan2(-fy, denom))
    # Up column for roll (camera Y axis)
    roll  = np.degrees(np.arctan2(R[0, 1], R[1, 1]))

    return float(yaw), float(pitch), float(roll)

def solve_head_pose_from_2d(image_points, img_w, img_h):
    # Approx 3D template (mm)
    object_points = np.array([
        [0.0,     0.0,    0.0],    # nose
        [0.0,   -330.0,  -65.0],   # chin
        [-225.0, 170.0, -135.0],   # left eye outer (subject's left)
        [225.0,  170.0, -135.0],   # right eye outer
        [-150.0,-150.0, -125.0],   # left mouth
        [150.0, -150.0, -125.0],   # right mouth
    ], dtype=np.float64)

    focal = img_w
    cam_matrix = np.array([[focal, 0, img_w/2],
                           [0, focal, img_h/2],
                           [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, None, None, None
    R, _ = cv2.Rodrigues(rvec)
    yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_ypr(R)
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, cam_matrix, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - image_points, axis=1)))
    return yaw_deg, pitch_deg, roll_deg, err

# -------- Robust camera/video open (Windows-safe) ----
API_MAP = {
    "auto": None,
    "any": None,
    "dshow": cv2.CAP_DSHOW,   # DirectShow (Windows)
    "msmf": cv2.CAP_MSMF,     # Media Foundation (Windows)
    "vfw": cv2.CAP_VFW,       # legacy
}

def _try_open_camera_once(index, api, timeout_open=3.0, timeout_first_frame=2.0, width=1280, height=720, fps=30):
    cap_holder = {}

    def worker():
        try:
            cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
            cap_holder["cap"] = cap
        except Exception:
            cap_holder["cap"] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start(); t.join(timeout_open)
    if t.is_alive():
        return None, "timeout_open"

    cap = cap_holder.get("cap")
    if cap is None or not cap.isOpened():
        try:
            if cap is not None: cap.release()
        except Exception:
            pass
        return None, "open_failed"

    # try basic settings (best-effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    t0 = time.time()
    while time.time() - t0 < timeout_first_frame:
        ok, _ = cap.read()
        if ok:
            return cap, None
        time.sleep(0.05)

    cap.release()
    return None, "no_frames"

def open_camera_robust(preferred_index, api_str="auto"):
    indices = [preferred_index] + [i for i in range(0, 4) if i != preferred_index]
    if api_str.lower() in ("auto", "any"):
        apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, None]
    else:
        apis = [API_MAP.get(api_str.lower(), None)]

    for api in apis:
        for idx in indices:
            cap, reason = _try_open_camera_once(idx, api)
            if cap is not None:
                used = "CAP_ANY" if api is None else {v:k for k,v in API_MAP.items()}.get(api, str(api))
                print(f"[camera] opened index {idx} via {used}")
                return cap
            # print(f"[camera] fail index {idx} api {api}: {reason}")
    raise RuntimeError("Could not open a camera. Try closing other apps, or run with --api dshow / --api msmf, or use --video <file>.")

# -------------------- Tracker ------------------------
class HeadTracker:
    def __init__(self, camera_index=0, api="auto", video_path=None,
                 min_confidence=0.5, osc_enabled=False, osc_host='127.0.0.1', osc_port=8053,
                 nogui=False):
        self.nogui = nogui

        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            print(f"[video] {video_path}")
        else:
            self.cap = open_camera_robust(camera_index, api_str=api)

        self.face_mesh = MP_FACE_MESH.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )

        # Filters
        self.f_yaw = OneEuroFilter(freq=60, min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
        self.f_pitch = OneEuroFilter(freq=60, min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
        self.f_roll = OneEuroFilter(freq=60, min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.f_x = OneEuroFilter(freq=60, min_cutoff=1.2, beta=0.02, d_cutoff=1.0)
        self.f_y = OneEuroFilter(freq=60, min_cutoff=1.2, beta=0.02, d_cutoff=1.0)

        self.calib = None
        self.osc = OSCSender(host=osc_host, port=osc_port, enabled=osc_enabled)

        self.last_state = {"yaw_deg":0.0,"pitch_deg":0.0,"roll_deg":0.0,"x_norm":0.5,"y_norm":0.5,"confidence":0.0,"reproj_error":None}

        self.calib_targets = [(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9),(0.5,0.5)]
        self.calib_samples = []
        self.calibrating = False
        self.current_target_idx = 0
        self.err_hist = deque(maxlen=30)

    @staticmethod
    def _landmarks_to_points(lms, img_w, img_h):
        pts = {}
        for name, idx in FACE_IDX.items():
            p = lms[idx]
            pts[name] = np.array([p.x * img_w, p.y * img_h], dtype=np.float64)
        return pts

    def _compute_pose(self, results, img_w, img_h):
        if not results.multi_face_landmarks:
            return None
        lms2d = results.multi_face_landmarks[0].landmark
        pts2d = self._landmarks_to_points(lms2d, img_w, img_h)
        names = ["nose_tip","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]
        image_points = np.array([pts2d[n] for n in names], dtype=np.float64)

        yaw_deg, pitch_deg, roll_deg, err = solve_head_pose_from_2d(image_points, img_w, img_h)
        if yaw_deg is None:
            return None

        yaw_s = float(self.f_yaw(yaw_deg))
        pitch_s = float(self.f_pitch(pitch_deg))
        roll_s = float(self.f_roll(roll_deg))

        self.err_hist.append(err if err is not None else 10.0)
        err_use = np.median(self.err_hist) if len(self.err_hist) else (err or 10.0)
        conf = float(np.clip(1.0 - (err_use / 20.0), 0.0, 1.0))

        if self.calib is not None:
            x_n, y_n = self.calib.map(yaw_s, pitch_s)
        else:
            x_n = (yaw_s + 25.0) / 50.0
            y_n = (pitch_s + 20.0) / 40.0
            x_n = float(np.clip(x_n, 0.0, 1.0)); y_n = float(np.clip(y_n, 0.0, 1.0))
        x_n = float(self.f_x(x_n)); y_n = float(self.f_y(y_n))

        self.last_state.update({"yaw_deg":yaw_s,"pitch_deg":pitch_s,"roll_deg":roll_s,
                                "x_norm":x_n,"y_norm":y_n,"confidence":conf,"reproj_error":err})
        return self.last_state

    def _draw_overlay(self, frame, state):
        h, w = frame.shape[:2]
        cx = int(state["x_norm"] * w); cy = int(state["y_norm"] * h)
        cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)
        cv2.circle(frame, (cx, cy), 16, (0,255,0), 1)
        y0, dy = 30, 24
        lines = [
            f"Yaw:   {state['yaw_deg']:6.1f} deg",
            f"Pitch: {state['pitch_deg']:6.1f} deg",
            f"Roll:  {state['roll_deg']:6.1f} deg",
            f"x,y:   {state['x_norm']:.3f}, {state['y_norm']:.3f}",
            f"Conf:  {state['confidence']:.2f}",
            f"Err(px): {state['reproj_error']:.2f}" if state['reproj_error'] is not None else "Err(px): --",
            "Keys: a=calibrate  s=save  l=load  o=OSC  q=quit"
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        if self.calibrating:
            tx = int(self.calib_targets[self.current_target_idx][0]*w)
            ty = int(self.calib_targets[self.current_target_idx][1]*h)
            cv2.circle(frame, (tx, ty), 10, (0,200,255), -1)
            cv2.circle(frame, (tx, ty), 20, (0,200,255), 2)
            cv2.putText(frame, f"CALIB {self.current_target_idx+1}/5: point head, press ENTER",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)

    def _handle_calib_keys(self, k, state):
        if k in (13, 10):  # Enter
            if self.calibrating and state is not None:
                tgt = self.calib_targets[self.current_target_idx]
                self.calib_samples.append((state["yaw_deg"], state["pitch_deg"], tgt))
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.calib_targets):
                    yaw_list = [s[0] for s in self.calib_samples]
                    pitch_list = [s[1] for s in self.calib_samples]
                    xy_targets = [s[2] for s in self.calib_samples]
                    self.calib = fit_calibration(yaw_list, pitch_list, xy_targets)
                    self.calibrating = False
                    self.calib_samples.clear()
                    self.current_target_idx = 0
                    print("Calibration complete.")
                else:
                    print(f"Captured {self.current_target_idx}/{len(self.calib_targets)}")
        elif k == ord('a'):
            self.calibrating = True; self.calib_samples.clear(); self.current_target_idx = 0
            print("Calibration started. Press ENTER at each dot.")
        elif k == ord('s'):
            self.save_calibration()
        elif k == ord('l'):
            self.load_calibration()
        elif k == ord('o'):
            self.osc.enabled = not self.osc.enabled
            print(f"OSC {'enabled' if self.osc.enabled else 'disabled'}.")

    def save_calibration(self, path=None):
        if self.calib is None: print("No calibration to save."); return
        path = path or os.path.expanduser("~/head_tracker_calibration.json")
        with open(path, "w", encoding="utf-8") as f: json.dump(self.calib.to_json(), f, indent=2)
        print(f"Saved calibration -> {path}")

    def load_calibration(self, path=None):
        path = path or os.path.expanduser("~/head_tracker_calibration.json")
        if not os.path.exists(path): print(f"No calibration at {path}"); return
        with open(path, "r", encoding="utf-8") as f: self.calib = CalibrationModel.from_json(json.load(f))
        print(f"Loaded calibration <- {path}")

    def _send_osc(self, state):
        self.osc.send("/head/ypr", float(state["yaw_deg"]), float(state["pitch_deg"]), float(state["roll_deg"]))
        self.osc.send("/gaze/norm", float(state["x_norm"]), float(state["y_norm"]), float(state["confidence"]))
        x_px = int(state["x_norm"] * SCREEN_W); y_px = int(state["y_norm"] * SCREEN_H)
        self.osc.send("/gaze/screen", x_px, y_px, float(state["confidence"]))

    def run(self):
        print("Head tracker running. 'a' calibrate, 'q' quit.")
        last_print = 0.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Read failed."); break
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            state = self._compute_pose(results, img_w, img_h)

            if self.osc.enabled and state is not None:
                self._send_osc(state)

            if self.nogui:
                now = time.time()
                if state is not None and (now - last_print) > 0.2:
                    last_print = now
                    print(f"yaw={state['yaw_deg']:+6.1f} pitch={state['pitch_deg']:+6.1f} "
                          f"x={state['x_norm']:.3f} y={state['y_norm']:.3f} conf={state['confidence']:.2f}")
            else:
                if state is not None: self._draw_overlay(frame, state)
                else: cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("Head Tracker (Head-as-Gaze)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27): break
                self._handle_calib_keys(k, state)

        self.cap.release()
        if not self.nogui: cv2.destroyAllWindows()

# ---------------------- CLI --------------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone Head-Directed Gaze Tracker (robust camera open)")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--api", type=str, default="auto", choices=list(API_MAP.keys()),
                    help="Camera backend: auto|dshow|msmf|vfw|any")
    ap.add_argument("--video", type=str, default=None, help="Use a video file instead of a camera")
    ap.add_argument("--osc", action="store_true", help="Enable OSC output")
    ap.add_argument("--osc_host", type=str, default="127.0.0.1", help="OSC host")
    ap.add_argument("--osc_port", type=int, default=8053, help="OSC port")
    ap.add_argument("--nogui", action="store_true", help="Run headless (no preview window)")
    args = ap.parse_args()

    tracker = HeadTracker(camera_index=args.camera, api=args.api, video_path=args.video,
                          osc_enabled=args.osc, osc_host=args.osc_host, osc_port=args.osc_port,
                          nogui=args.nogui)
    tracker.run()

if __name__ == "__main__":
    main()
