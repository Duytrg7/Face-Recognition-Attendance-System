"""Tiện ích dùng chung cho bộ test M1 (chạy trên Raspberry Pi 4)."""
import json
import os
import re
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(LOG_DIR, exist_ok=True)

CAMERA_PATH = "/dev/video0"
FRAME_W, FRAME_H = 640, 480

YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
FAS_MODELS = [  # (file, crop scale) — MiniFASNet từ Silent-Face-Anti-Spoofing
    (os.path.join(MODEL_DIR, "2.7_80x80_MiniFASNetV2.onnx"), 2.7),
    (os.path.join(MODEL_DIR, "4_0_0_80x80_MiniFASNetV1SE.onnx"), 4.0),
]


# ---------------------------------------------------------------- logging
def save_json(name, data):
    path = os.path.join(LOG_DIR, f"{name}.json")
    data = {"test": name, "time": datetime.now().isoformat(timespec="seconds"), **data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[LOG] Đã lưu {path}")
    return path


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


# ---------------------------------------------------------------- Pi health
def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except OSError:
        return None


def cpu_freq_mhz():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read()) / 1000.0
    except OSError:
        return None


def throttled():
    """vcgencmd get_throttled -> hex string (0x0 = ok)."""
    out = run("vcgencmd get_throttled")
    m = re.search(r"0x[0-9a-fA-F]+", out)
    return m.group(0) if m else out


# ---------------------------------------------------------------- camera
def open_camera(path=CAMERA_PATH, w=FRAME_W, h=FRAME_H):
    """Mở C270 giống ĐA1: V4L2 + MJPG + buffer 1."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {path}")
    for _ in range(15):  # warm-up
        cap.read()
    return cap


def v4l2_controls(dev=CAMERA_PATH):
    """Trả về dict {tên control: giá trị hiện tại} từ v4l2-ctl."""
    out = run(f"v4l2-ctl -d {dev} --list-ctrls")
    ctrls = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\).*?value=(-?\d+)", line)
        if m:
            ctrls[m.group(1)] = int(m.group(3))
    return ctrls, out


def lock_camera(dev=CAMERA_PATH, exposure=None, wb_temp=4600):
    """Khoá exposure + white balance. Hỗ trợ cả tên control kernel cũ và mới.
    exposure=None -> giữ giá trị exposure hiện tại (đọc lúc auto đã ổn định)."""
    ctrls, _ = v4l2_controls(dev)
    cmds = []
    if "auto_exposure" in ctrls:            # kernel mới: 1=Manual, 3=Aperture Priority
        cmds.append("auto_exposure=1")
        exp_name = "exposure_time_absolute"
    elif "exposure_auto" in ctrls:          # kernel cũ
        cmds.append("exposure_auto=1")
        exp_name = "exposure_absolute"
    else:
        exp_name = None
    if exp_name:
        val = exposure if exposure is not None else ctrls.get(exp_name, 250)
        cmds.append(f"{exp_name}={val}")
    if "exposure_dynamic_framerate" in ctrls:
        cmds.append("exposure_dynamic_framerate=0")
    if "white_balance_automatic" in ctrls:
        cmds.append("white_balance_automatic=0")
    elif "white_balance_temperature_auto" in ctrls:
        cmds.append("white_balance_temperature_auto=0")
    if "white_balance_temperature" in ctrls:
        cmds.append(f"white_balance_temperature={wb_temp}")
    for c in cmds:  # đặt từng cái một: auto phải tắt trước khi đặt giá trị tay
        run(f"v4l2-ctl -d {dev} --set-ctrl={c}")
    return cmds


def unlock_camera(dev=CAMERA_PATH):
    ctrls, _ = v4l2_controls(dev)
    if "auto_exposure" in ctrls:
        run(f"v4l2-ctl -d {dev} --set-ctrl=auto_exposure=3")
    elif "exposure_auto" in ctrls:
        run(f"v4l2-ctl -d {dev} --set-ctrl=exposure_auto=3")
    if "white_balance_automatic" in ctrls:
        run(f"v4l2-ctl -d {dev} --set-ctrl=white_balance_automatic=1")
    elif "white_balance_temperature_auto" in ctrls:
        run(f"v4l2-ctl -d {dev} --set-ctrl=white_balance_temperature_auto=1")


# ---------------------------------------------------------------- stats
def stats_ms(samples):
    if not samples:
        return None
    a = np.array(samples) * 1000.0
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2)}


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t


# ---------------------------------------------------------------- models
class YuNet:
    def __init__(self, path=YUNET_PATH, score=0.8):
        self.det = cv2.FaceDetectorYN.create(path, "", (320, 320), score, 0.3, 5000)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        return [] if faces is None else faces  # mỗi hàng: x,y,w,h, 5 landmark (10), score


class SFace:
    def __init__(self, path=SFACE_PATH):
        self.rec = cv2.FaceRecognizerSF.create(path, "")

    def embed(self, bgr, yunet_row):
        aligned = self.rec.alignCrop(bgr, yunet_row)
        return self.rec.feature(aligned)


class Liveness:
    """MiniFASNet x2 (crop 2.7x và 4.0x quanh bbox), cộng xác suất như repo gốc.
    Trả về điểm real trong [0,1]. Lưu ý: chưa fine-tune — chỉ dùng để đo tốc độ ở M1."""

    def __init__(self, models=FAS_MODELS):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sessions = [(ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]), s)
                         for p, s in models]

    @staticmethod
    def crop(img, bbox, scale, out=80):
        src_h, src_w = img.shape[:2]
        x, y, bw, bh = bbox
        scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = x + bw / 2, y + bh / 2
        x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if x1 < 0: x2 -= x1; x1 = 0
        if y1 < 0: y2 -= y1; y1 = 0
        if x2 > src_w - 1: x1 -= x2 - src_w + 1; x2 = src_w - 1
        if y2 > src_h - 1: y1 -= y2 - src_h + 1; y2 = src_h - 1
        patch = img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
        return cv2.resize(patch, (out, out))

    @staticmethod
    def margin_ok(img, bbox, scale=4.0):
        """True nếu vùng crop scale x bbox nằm trọn trong khung (không bị ép nhỏ)."""
        h, w = img.shape[:2]
        return bbox[2] * scale <= w - 1 and bbox[3] * scale <= h - 1

    def score(self, bgr, bbox):
        prob = np.zeros(3, dtype=np.float32)
        for sess, s in self.sessions:
            x = self.crop(bgr, bbox, s).transpose(2, 0, 1)[None].astype(np.float32)
            prob += sess.run(None, {"input": x})[0][0]
        return float(prob[1] / len(self.sessions))


def have(path):
    return os.path.exists(path) and os.path.getsize(path) > 10_000
