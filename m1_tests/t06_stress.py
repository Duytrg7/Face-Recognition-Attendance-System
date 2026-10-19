"""T06 — Chạy pipeline đầy đủ liên tục, ghi nhiệt độ / xung nhịp / throttling / tốc độ suy luận.

Kiến trúc giống thiết kế M1: grabber_thread luôn giữ frame mới nhất; vòng suy luận lấy frame
mới nhất -> detect -> liveness -> embedding. Ghi CSV mỗi 10 s vào logs/t06_stress.csv.

Chạy: python t06_stress.py [--minutes 15] [--detector yunet|hog] [--embedder dlib|sface]
Đứng trước camera (hoặc dựng ảnh/điện thoại có mặt người) để pipeline chạy đủ các bước.
Nên chạy 2 lần: VNC tắt (hiển thị qua HDMI capture) và VNC bật, để so sánh.
"""
import argparse
import csv
import os
import threading
import time

import cv2

from common import (LOG_DIR, SFACE_PATH, YUNET_PATH, Liveness, SFace, YuNet, cpu_freq_mhz,
                    cpu_temp, have, open_camera, run, save_json, throttled)

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=15)
ap.add_argument("--detector", choices=["yunet", "hog"], default="yunet")
ap.add_argument("--embedder", choices=["dlib", "sface"], default="dlib")
ap.add_argument("--tag", default="", help="nhãn tuỳ ý, vd: vnc_on / vnc_off")
args = ap.parse_args()
if args.detector == "yunet" and not have(YUNET_PATH):
    print("Thiếu YuNet -> dùng HOG"); args.detector = "hog"
if args.embedder == "sface" and not (have(SFACE_PATH) and args.detector == "yunet"):
    print("SFace cần YuNet + model -> dùng dlib"); args.embedder = "dlib"

import face_recognition  # noqa: E402

cap = open_camera()
latest = {"frame": None, "n": 0}
stop = False


def grabber():
    while not stop:
        ok, f = cap.read()
        if ok:
            latest["frame"] = f
            latest["n"] += 1


threading.Thread(target=grabber, daemon=True).start()
yn = YuNet() if args.detector == "yunet" else None
sf = SFace() if args.embedder == "sface" else None
live = Liveness()

tag = f"_{args.tag}" if args.tag else ""
csv_path = os.path.join(LOG_DIR, f"t06_stress_{args.detector}_{args.embedder}{tag}.csv")
fcsv = open(csv_path, "w", newline="")
w = csv.writer(fcsv)
w.writerow(["t_s", "temp_c", "freq_mhz", "throttled", "camera_fps", "inference_per_s",
            "faces_rate", "mean_latency_ms"])

t_start = time.time(); t_win = t_start; n_inf = n_face = 0; lat_sum = 0.0; cam_n0 = 0
rows = []
try:
    while time.time() - t_start < args.minutes * 60:
        f = latest["frame"]
        if f is None:
            time.sleep(0.01); continue
        t0 = time.perf_counter()
        if yn is not None:
            faces = yn.detect(f)
            bbox = [int(v) for v in faces[0][:4]] if len(faces) else None
            row = faces[0] if len(faces) else None
        else:
            rgb_s = cv2.cvtColor(cv2.resize(f, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            b = face_recognition.face_locations(rgb_s, 0, "hog")
            bbox = None
            if b:
                t_, r_, b_, l_ = [v * 2 for v in b[0]]
                bbox = [l_, t_, r_ - l_, b_ - t_]
        if bbox is not None:
            n_face += 1
            score = live.score(f, bbox)
            if sf is not None:
                sf.embed(f, row)
            else:
                x, y, bw, bh = bbox
                face_recognition.face_encodings(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                                [(y, x + bw, y + bh, x)], 1, "small")
        lat_sum += time.perf_counter() - t0
        n_inf += 1
        now = time.time()
        if now - t_win >= 10:
            dt = now - t_win
            r = [round(now - t_start), cpu_temp(), cpu_freq_mhz(), throttled(),
                 round((latest["n"] - cam_n0) / dt, 1), round(n_inf / dt, 2),
                 round(n_face / max(1, n_inf), 2), round(1000 * lat_sum / max(1, n_inf), 1)]
            w.writerow(r); fcsv.flush(); rows.append(r)
            print(f"t={r[0]:>5}s temp={r[1]}°C freq={r[2]}MHz thr={r[3]} cam={r[4]}fps "
                  f"inf={r[5]}/s faces={r[6]} lat={r[7]}ms")
            t_win = now; n_inf = n_face = 0; lat_sum = 0.0; cam_n0 = latest["n"]
except KeyboardInterrupt:
    print("Dừng sớm.")
finally:
    stop = True
    time.sleep(0.2)
    cap.release()
    fcsv.close()

summary = {"detector": args.detector, "embedder": args.embedder, "tag": args.tag,
           "minutes": round((time.time() - t_start) / 60, 1), "csv": csv_path,
           "vnc_running": run("pgrep -l -f 'vnc' | head -3") or "no"}
if rows:
    summary.update({"temp_max_c": max(r[1] for r in rows if r[1] is not None),
                    "inference_per_s_mean": round(sum(r[5] for r in rows) / len(rows), 2),
                    "camera_fps_mean": round(sum(r[4] for r in rows) / len(rows), 1),
                    "latency_ms_mean": round(sum(r[7] for r in rows) / len(rows), 1),
                    "throttled_end": throttled()})
print(summary)
save_json(f"t06_stress_{args.detector}_{args.embedder}{tag}", summary)
