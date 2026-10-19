"""T04 — Đo độ trễ từng khối xử lý trên Pi (detector, embedding, liveness).

Bước 1: chụp N frame có 1 người đứng trước camera ~1 m (hoặc dùng --images <thư mục>).
Bước 2: chạy từng khối trên cùng tập frame, đo thời gian (mean/p50/p95) và tỉ lệ tìm thấy mặt.

So sánh:
  HOG  @1/4 (đúng như ĐA1: CV_SCALER=4, upsample=0), @1/2, @1/1
  YuNet @640x480 và @320x240
  dlib encoding trên frame 1/4 (như ĐA1) và trên frame gốc
  SFace (alignCrop + feature)
  MiniFASNet x2 (liveness, ONNX Runtime)
Cuối cùng ước lượng thông lượng 3 cấu hình pipeline: (a) ĐA1, (b) YuNet+dlib+liveness,
(c) YuNet+SFace+liveness.

Chạy: python t04_bench_models.py [--frames 60] [--images DIR] [--keep]
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from common import (LOG_DIR, SFACE_PATH, YUNET_PATH, Liveness, SFace, Timer, YuNet,
                    cpu_temp, have, open_camera, save_json, stats_ms, throttled)

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--images", default=None)
ap.add_argument("--keep", action="store_true", help="lưu frame đã chụp vào logs/t04_frames")
args = ap.parse_args()

# ---------------------------------------------------------------- frames
if args.images:
    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
    def fit(f):  # giữ tỉ lệ, đệm viền đen về 640x480
        if f.shape[:2] == (480, 640):
            return f
        s = min(640 / f.shape[1], 480 / f.shape[0])
        r = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        out = np.zeros((480, 640, 3), np.uint8)
        y, x = (480 - r.shape[0]) // 2, (640 - r.shape[1]) // 2
        out[y:y + r.shape[0], x:x + r.shape[1]] = r
        return out
    frames = [fit(f) for f in (cv2.imread(p) for p in paths) if f is not None]
else:
    cap = open_camera()
    input(f"Đứng trước camera ~1 m, nhìn thẳng. Nhấn Enter để chụp {args.frames} frame...")
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if ok:
            frames.append(f)
        time.sleep(0.05)
    cap.release()
    if args.keep:
        d = os.path.join(LOG_DIR, "t04_frames"); os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(d, f"{i:03d}.jpg"), f)
print(f"Số frame: {len(frames)}  |  nhiệt độ đầu: {cpu_temp()} °C")

R = {"n_frames": len(frames), "temp_start_c": cpu_temp()}

# ---------------------------------------------------------------- HOG
import face_recognition  # noqa: E402

hog_boxes_full = []  # box (top,right,bottom,left) trên frame gốc, dùng cho dlib full-res
for scale in (4, 2, 1):
    times, found = [], 0
    for f in frames:
        small = cv2.resize(f, (0, 0), fx=1 / scale, fy=1 / scale) if scale > 1 else f
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        with Timer() as t:
            boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=0, model="hog")
        times.append(t.dt)
        found += bool(boxes)
        if scale == 1:
            hog_boxes_full.append(boxes)
    R[f"hog_1/{scale}"] = {**stats_ms(times), "detect_rate": round(found / len(frames), 3)}
    print(f"HOG 1/{scale}: {R[f'hog_1/{scale}']}")

# ---------------------------------------------------------------- dlib encoding
# (a) như ĐA1: encoding trên frame 1/4
times = []
for f in frames:
    rgb = cv2.cvtColor(cv2.resize(f, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, 0, "hog")
    if boxes:
        with Timer() as t:
            face_recognition.face_encodings(rgb, boxes[:1], num_jitters=1, model="small")
        times.append(t.dt)
R["dlib_encode_on_1/4"] = stats_ms(times)
# (b) encoding trên frame gốc (box HOG full-res)
times = []
for f, boxes in zip(frames, hog_boxes_full):
    if boxes:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        with Timer() as t:
            face_recognition.face_encodings(rgb, boxes[:1], num_jitters=1, model="small")
        times.append(t.dt)
R["dlib_encode_full"] = stats_ms(times)
print("dlib encode 1/4:", R["dlib_encode_on_1/4"], "\ndlib encode full:", R["dlib_encode_full"])

# ---------------------------------------------------------------- YuNet
yunet_rows = []
if have(YUNET_PATH):
    yn = YuNet()
    for size in ((640, 480), (320, 240)):
        times, found, face_px = [], 0, []
        for f in frames:
            img = cv2.resize(f, size) if size != (640, 480) else f
            with Timer() as t:
                faces = yn.detect(img)
            times.append(t.dt)
            if len(faces):
                found += 1
                face_px.append(float(faces[0][2]) * 640 / size[0])
            if size == (640, 480):
                yunet_rows.append(faces[0] if len(faces) else None)
        key = f"yunet_{size[0]}x{size[1]}"
        R[key] = {**stats_ms(times), "detect_rate": round(found / len(frames), 3),
                  "face_width_px_mean": round(float(np.mean(face_px)), 1) if face_px else None}
        print(key, R[key])
    # dlib encoding với box YuNet (đổi x,y,w,h -> top,right,bottom,left)
    times = []
    for f, row in zip(frames, yunet_rows):
        if row is None:
            continue
        x, y, w, h = [int(v) for v in row[:4]]
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        with Timer() as t:
            face_recognition.face_encodings(rgb, [(y, x + w, y + h, x)], num_jitters=1, model="small")
        times.append(t.dt)
    R["dlib_encode_full_yunet_box"] = stats_ms(times)
else:
    R["yunet"] = "SKIPPED — thiếu models/face_detection_yunet_2023mar.onnx (chạy download_models.sh)"
    print(R["yunet"])

# ---------------------------------------------------------------- SFace
if have(SFACE_PATH) and yunet_rows:
    sf = SFace()
    times = []
    for f, row in zip(frames, yunet_rows):
        if row is not None:
            with Timer() as t:
                sf.embed(f, row)
            times.append(t.dt)
    R["sface_embed"] = stats_ms(times)
    print("SFace:", R["sface_embed"])
else:
    R["sface_embed"] = "SKIPPED — thiếu SFace hoặc YuNet"

# ---------------------------------------------------------------- Liveness
live = Liveness()
times, scores, margin_ok = [], [], 0
for i, f in enumerate(frames):
    if yunet_rows and yunet_rows[i] is not None:
        bbox = [int(v) for v in yunet_rows[i][:4]]
    elif hog_boxes_full[i]:
        t_, r_, b_, l_ = hog_boxes_full[i][0]
        bbox = [l_, t_, r_ - l_, b_ - t_]
    else:
        continue
    margin_ok += Liveness.margin_ok(f, bbox, 4.0)
    with Timer() as t:
        s = live.score(f, bbox)
    times.append(t.dt)
    scores.append(s)
R["liveness_minifasnet_x2"] = {**(stats_ms(times) or {}),
                               "real_score_mean_pretrained": round(float(np.mean(scores)), 3) if scores else None,
                               "crop4x_inside_frame_rate": round(margin_ok / max(1, len(times)), 3)}
print("Liveness:", R["liveness_minifasnet_x2"])


# ---------------------------------------------------------------- pipeline estimate
def m(key):
    v = R.get(key)
    return v["mean_ms"] if isinstance(v, dict) and v and "mean_ms" in v else None


def est(parts):
    if any(p is None for p in parts):
        return None
    tot = sum(parts)
    return {"latency_ms": round(tot, 1), "inference_per_s": round(1000 / tot, 2)}


R["pipeline_estimate"] = {
    "a_DA1_hog1/4_dlib1/4": est([m("hog_1/4"), m("dlib_encode_on_1/4")]),
    "b_yunet_dlib_liveness": est([m("yunet_640x480"), m("dlib_encode_full_yunet_box"), m("liveness_minifasnet_x2")]),
    "c_yunet_sface_liveness": est([m("yunet_640x480"), m("sface_embed"), m("liveness_minifasnet_x2")]),
    "hog1/2_dlibfull_liveness": est([m("hog_1/2"), m("dlib_encode_full"), m("liveness_minifasnet_x2")]),
}
R["temp_end_c"] = cpu_temp()
R["throttled"] = throttled()
print("\nƯớc lượng pipeline (tuần tự, 1 mặt):")
for k, v in R["pipeline_estimate"].items():
    print(f"  {k:28} {v}")
save_json("t04_bench_models", R)
