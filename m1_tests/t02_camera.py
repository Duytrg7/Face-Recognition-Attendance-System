"""T02 — Khối camera Logitech C270.
1. Liệt kê format/control của camera.
2. Đo FPS đọc thô (không xử lý) ở chế độ AUTO và LOCKED (exposure + WB cố định).
3. Đo độ ổn định độ sáng (std của mean gray qua các frame) ở 2 chế độ.
4. Lưu ảnh mẫu của từng chế độ vào logs/.

Chạy: python t02_camera.py [--seconds 10] [--exposure 250]
Gợi ý: trong lúc đo, thử đi qua lại / bật tắt đèn để thấy AUTO dao động.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import (CAMERA_PATH, LOG_DIR, lock_camera, open_camera, run, save_json,
                    unlock_camera, v4l2_controls)

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--exposure", type=int, default=None, help="giá trị exposure khi khoá (mặc định: giữ giá trị auto hiện tại)")
args = ap.parse_args()

result = {"formats": run(f"v4l2-ctl -d {CAMERA_PATH} --list-formats-ext")}


def measure(cap, label):
    n, means, t0 = 0, [], time.perf_counter()
    frame = None
    while time.perf_counter() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        means.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    dt = time.perf_counter() - t0
    if frame is not None:
        cv2.imwrite(os.path.join(LOG_DIR, f"t02_sample_{label}.jpg"), frame)
    r = {"fps": round(n / dt, 2), "frames": n, "shape": None if frame is None else list(frame.shape),
         "brightness_mean": round(float(np.mean(means)), 2) if means else None,
         "brightness_std": round(float(np.std(means)), 2) if means else None}
    print(f"[{label}] FPS={r['fps']}  brightness mean={r['brightness_mean']} std={r['brightness_std']}")
    return r


cap = open_camera()
print("Camera mở OK:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
      "fourcc=MJPG")

unlock_camera()
time.sleep(2)
ctrls_auto, raw_auto = v4l2_controls()
result["controls_auto"] = ctrls_auto
print(f"\n== AUTO ({args.seconds}s) ==")
result["auto"] = measure(cap, "auto")

cmds = lock_camera(exposure=args.exposure)
time.sleep(1)
ctrls_locked, _ = v4l2_controls()
result["lock_commands"] = cmds
result["controls_locked"] = ctrls_locked
print(f"\n== LOCKED ({args.seconds}s) == lệnh: {cmds}")
result["locked"] = measure(cap, "locked")

cap.release()
result["controls_raw"] = raw_auto
print("\nCamera vẫn ở chế độ LOCKED. Chạy `python -c \"import common; common.unlock_camera()\"` để trả về auto.")
save_json("t02_camera", result)
