"""T05 — Tỉ lệ phát hiện mặt theo khoảng cách: HOG@1/4 (ĐA1) vs HOG@1/2 vs YuNet.

Mục đích: kiểm tra giả thuyết "CV_SCALER=4 làm mặt quá nhỏ ở khoảng cách >= 1 m"
(HOG của dlib cần mặt ~80 px trong ảnh đưa vào; ở 1/4 tức ~320 px trên frame gốc).

Chạy: python t05_detect_distance.py [--seconds 8]
Làm lần lượt: 0.5 m, 1.0 m, 1.5 m (đo bằng thước), nhìn thẳng, đủ sáng.
"""
import argparse
import time

import cv2

from common import YUNET_PATH, YuNet, have, open_camera, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=8)
ap.add_argument("--distances", default="0.5,1.0,1.5")
args = ap.parse_args()

import face_recognition  # noqa: E402

yn = YuNet() if have(YUNET_PATH) else None
cap = open_camera()
R = {"seconds_per_distance": args.seconds, "results": {}}

for d in args.distances.split(","):
    input(f"\n>>> Đứng cách camera {d} m, nhìn thẳng. Nhấn Enter để bắt đầu...")
    cnt = {"frames": 0, "hog_1/4": 0, "hog_1/2": 0, "yunet": 0}
    widths = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        ok, f = cap.read()
        if not ok:
            continue
        cnt["frames"] += 1
        for key, s in (("hog_1/4", 4), ("hog_1/2", 2)):
            small = cv2.cvtColor(cv2.resize(f, (0, 0), fx=1 / s, fy=1 / s), cv2.COLOR_BGR2RGB)
            if face_recognition.face_locations(small, 0, "hog"):
                cnt[key] += 1
        if yn is not None:
            faces = yn.detect(f)
            if len(faces):
                cnt["yunet"] += 1
                widths.append(float(faces[0][2]))
    res = {k: round(v / max(1, cnt["frames"]), 3) for k, v in cnt.items() if k != "frames"}
    res["frames"] = cnt["frames"]
    res["face_width_px_mean"] = round(sum(widths) / len(widths), 1) if widths else None
    if yn is None:
        res["yunet"] = "SKIPPED"
    R["results"][d] = res
    print(f"{d} m -> {res}")

cap.release()
save_json("t05_detect_distance", R)
