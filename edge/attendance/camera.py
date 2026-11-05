"""Nguồn khung hình: webcam C270 (V4L2, MJPG, khoá thông số) hoặc video/thư mục ảnh để chạy thử."""
import glob
import os
import re
import subprocess
import time

import cv2
import numpy as np


def _run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return ""


def v4l2_controls(dev):
    out = _run(f"v4l2-ctl -d {dev} --list-ctrls")
    ctrls = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\).*?value=(-?\d+)", line)
        if m:
            ctrls[m.group(1)] = int(m.group(3))
    return ctrls


def lock_camera(dev, exposure, wb, gain):
    """Khoá exposure, white balance, gain. Hỗ trợ tên control kernel cũ và mới."""
    c = v4l2_controls(dev)
    cmds = []
    if "auto_exposure" in c:
        cmds += ["auto_exposure=1", f"exposure_time_absolute={exposure}"]
    elif "exposure_auto" in c:
        cmds += ["exposure_auto=1", f"exposure_absolute={exposure}"]
    if "exposure_dynamic_framerate" in c:
        cmds.append("exposure_dynamic_framerate=0")
    if "white_balance_automatic" in c:
        cmds.append("white_balance_automatic=0")
    elif "white_balance_temperature_auto" in c:
        cmds.append("white_balance_temperature_auto=0")
    if "white_balance_temperature" in c:
        cmds.append(f"white_balance_temperature={wb}")
    if "gain" in c and gain is not None:
        cmds.append(f"gain={gain}")
    for x in cmds:
        _run(f"v4l2-ctl -d {dev} --set-ctrl={x}")
    return cmds


class Source:
    """read() -> (ok, frame BGR đúng kích thước cấu hình)."""

    def __init__(self, cam_cfg):
        self.cfg = cam_cfg
        self.size = (cam_cfg["width"], cam_cfg["height"])
        src = str(cam_cfg["source"])
        self.kind, self.cap, self.files, self.i = "camera", None, [], 0
        self.period = 1.0 / 15
        self.t_last = 0.0
        if os.path.isdir(src):
            self.kind = "dir"
            self.files = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))
            if not self.files:
                raise RuntimeError(f"Thư mục {src} không có ảnh")
        elif os.path.isfile(src) and not src.startswith("/dev/"):
            self.kind = "video"
            self.cap = cv2.VideoCapture(src)
        else:
            dev = src if src.startswith("/dev/") else int(src)
            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self.cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise RuntimeError(f"Không mở được camera {src}")
            for _ in range(10):
                self.cap.read()
            if cam_cfg.get("lock") and isinstance(dev, str):
                self.locked = lock_camera(dev, cam_cfg["exposure"], cam_cfg["white_balance"], cam_cfg.get("gain"))
        if self.cap is not None and self.kind == "video" and not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video {src}")

    def _fit(self, f):
        if f is None:
            return None
        if (f.shape[1], f.shape[0]) != self.size:
            s = min(self.size[0] / f.shape[1], self.size[1] / f.shape[0])
            r = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
            out = np.zeros((self.size[1], self.size[0], 3), np.uint8)
            y, x = (self.size[1] - r.shape[0]) // 2, (self.size[0] - r.shape[1]) // 2
            out[y:y + r.shape[0], x:x + r.shape[1]] = r
            f = out
        return f

    def _pace(self):
        dt = time.monotonic() - self.t_last
        if dt < self.period:
            time.sleep(self.period - dt)
        self.t_last = time.monotonic()

    def read(self):
        if self.kind == "camera":
            ok, f = self.cap.read()
            return ok, self._fit(f) if ok else None
        self._pace()
        if self.kind == "dir":
            f = cv2.imread(self.files[self.i % len(self.files)])
            self.i += 1
            return f is not None, self._fit(f)
        ok, f = self.cap.read()
        if not ok:  # lặp lại video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, self._fit(f) if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
