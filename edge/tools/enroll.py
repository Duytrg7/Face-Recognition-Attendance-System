"""Đăng ký khuôn mặt trên Pi (bản M2, chưa có web quản trị).

Mỗi mẫu phải đạt kiểm tra chất lượng VÀ liveness. Lưu embedding cho cả SFace và dlib (nếu có mô hình),
nên đổi embedder sau này không cần đăng ký lại. Ảnh crop (lề 2x) lưu ở data/enroll/<mã>/ để tạo lại embedding.

Ví dụ (tắt ứng dụng chấm công trước vì camera chỉ mở được bởi 1 chương trình):
  python tools/enroll.py --code NV001 --name "Trương Sĩ Duy"                  # chụp từ camera, 8 mẫu
  python tools/enroll.py --code NV002 --name "Nguyễn Đào Duy" --from-dir ../dataset/nguyen_dao_duy
  python tools/enroll.py --list
  python tools/enroll.py --delete NV002
  python tools/enroll.py --reembed            # tạo lại embedding từ ảnh crop đã lưu
"""
import argparse
import glob
import os
import select
import sys
import time
import unicodedata

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attendance import config as config_mod  # noqa: E402
from attendance.camera import Source  # noqa: E402
from attendance.db import DB  # noqa: E402
from attendance.vision import (DlibEmbedder, Liveness, SFaceEmbedder, check_quality,  # noqa: E402
                               make_detector)

# Kế hoạch tư thế cho 8 mẫu (lặp lại nếu --samples khác 8)
POSES = [
    "Nhìn thẳng vào camera, cách khoảng 50cm",
    "Quay đầu nhẹ sang TRÁI (khoảng 15 độ)",
    "Quay đầu nhẹ sang TRÁI thêm một chút (khoảng 25 độ)",
    "Quay đầu nhẹ sang PHẢI (khoảng 15 độ)",
    "Quay đầu nhẹ sang PHẢI thêm một chút (khoảng 25 độ)",
    "Ngẩng đầu nhẹ lên",
    "Cúi đầu nhẹ xuống",
    "Nghiêng đầu nhẹ (tai lệch về một bên vai)",
]
POSE_STRAIGHT = {0}  # tư thế nhìn thẳng dùng ngưỡng góc chuẩn


def no_accent(t):
    """cv2.putText không vẽ được tiếng Việt có dấu -> bỏ dấu."""
    t = t.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def enter_pressed():
    """Enter ở terminal (không chặn vòng lặp)."""
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return False
    return bool(r) and sys.stdin.readline() != ""


def load_embedders(models_dir, detector_name):
    embs = []
    if detector_name == "yunet" and os.path.exists(os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")):
        embs.append(SFaceEmbedder(models_dir))
    try:
        embs.append(DlibEmbedder(models_dir))
    except Exception as e:  # noqa: BLE001
        print(f"[!] Bỏ qua dlib: {e}")
    if not embs:
        sys.exit("Không có embedder nào dùng được")
    return embs


def crop_margin(frame, face, k=2.0):
    H, W = frame.shape[:2]
    x, y, w, h = face.bbox
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * k
    x1, y1, x2, y2 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)), int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
    return frame[y1:y2, x1:x2]


def save_sample(db, eid, code, frame, face, embs, out_dir, idx, source):
    for e in embs:
        db.add_embedding(eid, e.model_version, e.embed(frame, face), source)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{idx:02d}.jpg"), crop_margin(frame, face))


def enroll_camera(cfg, db, det, live, embs, code, name, n):
    """Mỗi mẫu: hiện tư thế cần làm -> người dùng chỉnh xong nhấn Enter -> chụp khung đầu tiên đạt
    chất lượng + liveness trong 3 giây. Không đạt thì báo lý do và chờ Enter lại (không bỏ qua mẫu)."""
    eid = db.add_employee(code, name)
    out_dir = os.path.join(cfg["paths"]["enroll_images"], code)
    src = Source(cfg["camera"])
    show = bool(os.environ.get("DISPLAY"))
    thr = cfg["liveness"]["threshold"]
    got, armed, t_armed = 0, False, 0.0
    status, color = "", (255, 255, 255)
    print(f"\nĐăng ký {code} – {name}: {n} mẫu. Đứng cách camera 30–50cm.")
    print("Mỗi mẫu: chỉnh tư thế theo hướng dẫn rồi nhấn ENTER (ở terminal hoặc cửa sổ camera). Ctrl+C / Esc để huỷ.")

    def announce():
        print(f"\n[{got + 1}/{n}] Tư thế: {POSES[got % len(POSES)]}  -> chỉnh xong nhấn ENTER")

    announce()
    try:
        while got < n:
            ok, f = src.read()
            if not ok:
                continue
            pose = POSES[got % len(POSES)]
            faces = det.detect(f)
            face = max(faces, key=lambda x: x.area) if faces else None
            key = -1
            if show:
                disp = f.copy()
                if face is not None:
                    x, y, w, h = [int(v) for v in face.bbox]
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 200, 0) if armed else (200, 200, 200), 2)
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 58), (0, 0, 0), -1)
                cv2.putText(disp, no_accent(f"Mau {got + 1}/{n}: {pose}"), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                line2 = "Dang chup..." if armed else (no_accent(status) if status else "Nhan ENTER de chup")
                cv2.putText(disp, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 255) if armed else color, 1, cv2.LINE_AA)
                cv2.imshow("Dang ky khuon mat", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            if not armed and (key in (13, 10, 32) or enter_pressed()):
                armed, t_armed = True, time.monotonic()
                status = ""
                for _ in range(3):  # bỏ khung cũ trong bộ đệm camera
                    src.read()
                continue
            if not armed:
                continue
            reason = "Không thấy mặt"
            if face is not None:
                q = dict(cfg["quality"])
                if (got % len(POSES)) not in POSE_STRAIGHT:  # cho phép quay/nghiêng ở tư thế phụ
                    q["yaw_ratio"] = [0.35, 2.8]
                okq, hint = check_quality(f, face, q)
                ls = live.score(f, face) if live else 1.0
                if not okq:
                    reason = hint
                elif ls < thr:
                    reason = f"Liveness thấp ({ls:.2f}) – kiểm tra ánh sáng"
                else:
                    got += 1
                    save_sample(db, eid, code, f, face, embs, out_dir, got, "camera")
                    db.commit()
                    print(f"   ✓ Đã chụp mẫu {got}/{n}  (liveness {ls:.2f}, mặt rộng {int(face.bbox[2])}px)")
                    status, color, armed = f"Da chup mau {got}", (0, 220, 0), False
                    if got < n:
                        announce()
                    continue
            if time.monotonic() - t_armed > 3:
                armed = False
                status, color = f"Chưa đạt: {reason}. Nhấn ENTER thử lại", (0, 0, 255)
                print(f"   ✗ Chưa đạt: {reason}. Chỉnh lại rồi nhấn ENTER")
        db.commit()
    except KeyboardInterrupt:
        print("\nĐã huỷ (các mẫu đã chụp vẫn được lưu).")
        db.commit()
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()
    print(f"\nXong: {got}/{n} mẫu, embedding: {', '.join(e.name for e in embs)}")


def enroll_dir(cfg, db, det, live, embs, code, name, folder):
    eid = db.add_employee(code, name)
    out_dir = os.path.join(cfg["paths"]["enroll_images"], code)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    q = dict(cfg["quality"])
    q["yaw_ratio"] = [0.4, 2.5]
    got = 0
    for p in files:
        f = cv2.imread(p)
        if f is None:
            continue
        if f.shape[1] != cfg["camera"]["width"]:
            s = cfg["camera"]["width"] / f.shape[1]
            f = cv2.resize(f, (cfg["camera"]["width"], int(f.shape[0] * s)))
        faces = det.detect(f)
        if not faces:
            print(f"  bỏ {os.path.basename(p)}: không thấy mặt"); continue
        face = max(faces, key=lambda x: x.area)
        okq, hint = check_quality(f, face, q)
        if not okq:
            print(f"  bỏ {os.path.basename(p)}: {hint}"); continue
        ls = live.score(f, face) if live else 1.0
        if ls < cfg["liveness"]["threshold"]:
            print(f"  bỏ {os.path.basename(p)}: liveness {ls:.2f} (có thể là ảnh chụp màn hình/ảnh in)"); continue
        got += 1
        save_sample(db, eid, code, f, face, embs, out_dir, got, "folder")
    db.commit()
    print(f"Xong {code}: {got}/{len(files)} ảnh được dùng")


def reembed(cfg, db, det, embs):
    for eid, code, name, *_ in db.employees():
        db.con.execute("DELETE FROM embeddings WHERE employee_id=?", (eid,))
        n = 0
        for p in sorted(glob.glob(os.path.join(cfg["paths"]["enroll_images"], code, "*.jpg"))):
            f = cv2.imread(p)
            faces = det.detect(f) if f is not None else []
            if faces:
                face = max(faces, key=lambda x: x.area)
                for e in embs:
                    db.add_embedding(eid, e.model_version, e.embed(f, face), "reembed")
                n += 1
        print(f"  {code} {name}: {n} ảnh")
    db.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code")
    ap.add_argument("--name")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--from-dir")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete")
    ap.add_argument("--reembed", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config_mod.load(a.config)
    db = DB(cfg["paths"]["db"])
    if a.list:
        rows = db.employees()
        print(f"{'Mã':8} {'Họ tên':28} {'SFace':>6} {'dlib':>6}")
        for _, code, name, ns, nd in rows:
            print(f"{code:8} {name:28} {ns:6} {nd:6}")
        return
    if a.delete:
        print("Đã xoá" if db.delete_employee(a.delete) else "Không tìm thấy", a.delete)
        return
    md = cfg["paths"]["models"]
    det = make_detector(cfg, md)
    embs = load_embedders(md, det.name)
    if a.reembed:
        reembed(cfg, db, det, embs)
        return
    if not (a.code and a.name):
        ap.error("cần --code và --name")
    lv = cfg["liveness"]
    live = Liveness(md, lv["models"], lv.get("crop_scale"), lv.get("threads", 2)) if lv["enabled"] else None
    if a.from_dir:
        enroll_dir(cfg, db, det, live, embs, a.code, a.name, a.from_dir)
    else:
        enroll_camera(cfg, db, det, live, embs, a.code, a.name, a.samples)


if __name__ == "__main__":
    main()