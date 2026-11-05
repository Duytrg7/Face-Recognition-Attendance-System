# Edge module – Đồ án 2, mốc M2

Module cốt lõi chạy trên Raspberry Pi 4: **camera C270 → YuNet → kiểm tra chất lượng → liveness MiniFASNet (ONNX) → embedding (SFace/dlib) → bỏ phiếu k/N → SQLite + outbox → kiosk**.
Kiến trúc **2 process** (thiết kế M1, mục 3.2): Process A (camera, kiosk, watchdog) và Process B (suy luận) trao đổi khung hình qua shared memory, nên luồng camera không bị khựng khi mô hình đang tính.

```
edge/
├── config.yaml              # toàn bộ tham số (camera, ngưỡng, k/N, embedder…)
├── attendance/
│   ├── app.py               # Process A: grabber + watchdog + kiosk / headless
│   ├── inference.py         # Process B: pipeline + quyết định + ghi DB
│   ├── vision.py            # YuNet/HOG, kiểm tra chất lượng, Liveness, SFace/dlib, Gallery
│   ├── decision.py          # track + máy trạng thái k/N
│   ├── db.py                # SQLite WAL: employees, embeddings, local_attendance, outbox
│   ├── shm.py               # bộ đệm khung hình 2 slot (shared memory)
│   ├── camera.py            # C270 (khoá exposure/WB/gain) hoặc video/thư mục ảnh để test
│   └── kiosk.py             # giao diện CustomTkinter
├── tools/
│   ├── enroll.py            # đăng ký khuôn mặt (camera hoặc thư mục ảnh)
│   ├── bench_lfw.py         # benchmark LFW: HOG+dlib / YuNet+dlib / YuNet+SFace
│   └── summarize.py         # tổng hợp số liệu các phiên thử -> logs/summary_m2.json
├── models/                  # MiniFASNet x2 (có sẵn) + YuNet, SFace (tải bằng download_models.sh)
└── systemd/attendance.service
```

## 1. Cài đặt trên Pi (một lần)

```bash
cd ~/face_attendance && git switch da2-dev          # repo ĐA1, nhánh ĐA2
# chép thư mục edge/ vào đây (cạnh scripts/, m1_tests/)
cd edge
source ~/face_rec/bin/activate                       # venv ĐA1 (đổi nếu khác)
pip install -r requirements.txt
cp ../m1_tests/models/face_*.onnx models/ 2>/dev/null || bash download_models.sh
python -c "import cv2, onnxruntime, customtkinter, yaml; print('OK', cv2.__version__)"
```

## 2. Chốt embedder bằng LFW (chạy trước, để chạy nền)

Khuyên chạy trên **laptop** (nhanh); trên Pi thì dùng `--folds 3` và chạy trong `tmux` (~40 phút):
```bash
python tools/bench_lfw.py --folds 10      # laptop: pip install opencv-python numpy dlib-bin face_recognition_models "setuptools<81"
python tools/bench_lfw.py --folds 3       # Pi
```
Kết quả: `logs/bench_lfw.json`, `logs/bench_lfw_roc.png`. Quy tắc chốt: nếu SFace kém dlib **không quá ~1 điểm %** accuracy thì chọn **SFace** (nhanh hơn 8 lần trên Pi); ngược lại chọn dlib với E=3. Sửa `embedder.name` trong `config.yaml`. Ngưỡng `match_threshold` có thể lấy từ cột `threshold` của kết quả.

## 3. Đăng ký 2 người

Tắt kiosk trước (camera chỉ mở được bởi 1 chương trình). Mỗi mẫu phải đạt chất lượng **và** liveness:
```bash
python tools/enroll.py --code NV001 --name "Trương Sĩ Duy"        # 12 mẫu, làm theo hướng dẫn tư thế
python tools/enroll.py --code NV002 --name "Nguyễn Đào Duy"
python tools/enroll.py --code NV002 --name "Nguyễn Đào Duy" --from-dir ../dataset/<thư_mục_ĐA1>   # (tuỳ chọn) dùng thêm ảnh ĐA1
python tools/enroll.py --list
```
Embedding được lưu cho **cả SFace và dlib**, đổi embedder không cần đăng ký lại. Ảnh crop ở `data/enroll/` (không đưa lên Git).

## 4. Chạy

```bash
python -m attendance.app                    # kiosk (xem qua HDMI capture / VNC); chạy từ SSH thì thêm DISPLAY=:0 ở đầu lệnh
python -m attendance.app --headless         # qua SSH, in kết quả ra terminal
```
Thanh trạng thái hiện: FPS camera, tốc độ suy luận thực tế, số mẫu đã đăng ký, số sự kiện chờ đồng bộ. Mỗi quyết định ghi vào `logs/decisions.csv`; ảnh giả mạo lưu ở `data/snapshots/`.
**Khi quay demo Vào → Ra liên tiếp**, đặt `decision.min_gap_min: 0.5` (mặc định 5 phút).

## 5. Thu số liệu cho báo cáo M2 (sơ bộ, 2 người)

Mỗi phiên chạy app với nhãn `--session`. Một **lượt** = bước vào khung, chờ kết quả, bước ra hẳn khỏi khung (để track kết thúc), rồi lặp lại.

| Phiên | Lệnh | Số lượt gợi ý |
|---|---|---|
| Mặt thật | `--session real:NV001` (rồi `real:NV002`) | 20 lượt/người, đổi khoảng cách 0,3–0,5m, góc nhẹ |
| Ảnh in | `--session print` | 10 lượt/người (A4 + 10×15) |
| Ảnh trên điện thoại | `--session screen` | 10 lượt/người, 2 mức sáng màn hình |
| Video phát lại | `--session video` | 10 lượt/người |
| Người lạ | `python tools/enroll.py --delete NV002` → `--session unknown` với người NV002 → đăng ký lại | 10 lượt |

```bash
python -m attendance.app --headless --session real:NV001
python tools/summarize.py          # -> logs/summary_m2.json: nhận diện đúng, BPCER, APCER theo kiểu, FAR người lạ, độ trễ
```

## 6. Kịch bản video demo (3–5 phút)

1. Giới thiệu phần cứng (Pi + C270 + capture card) và kiến trúc 2 process.
2. Đăng ký nhanh một người bằng `tools/enroll.py`.
3. Chấm công **Vào**, bước ra, chấm **Ra** (min_gap 0.5 phút).
4. Ảnh in → **KHÔNG HỢP LỆ**; ảnh trên điện thoại → chặn; video điện thoại → chặn.
5. Người chưa đăng ký → "Chưa đăng ký".
6. Kill Process B (`pkill -f spawn_main`) → watchdog tự khởi động lại, camera vẫn chạy.
7. Mở `logs/decisions.csv` / `summary_m2.json`, `sqlite3 data/attendance.db "select * from outbox"`.

## 7. Đưa lên Git

Thêm vào `.gitignore` của repo:
```
edge/data/
edge/logs/
edge/models/face_detection_yunet_*.onnx
edge/models/face_recognition_sface_*.onnx
```
```bash
git add edge .gitignore && git status        # kiểm tra KHÔNG có data/, logs/, ảnh .jpg
git commit -m "M2: edge module 2 process, pipeline liveness + nhận diện, kiosk"
git push && git tag m2 && git push origin m2
```

## 8. Sự cố thường gặp

- `Không mở được camera`: tắt GUI ĐA1 / kiosk khác (`sudo fuser -v /dev/video0`).
- Liveness luôn thấp với mặt thật: kiểm tra ánh sáng; xem `liveness` trong `logs/decisions.csv`; tạm hạ `liveness.threshold`.
- Không nhận ra người đã đăng ký: kiểm tra `tools/enroll.py --list` có embedding cho embedder đang dùng; nới `match_threshold`.
- Watchdog khởi động lại liên tục: xem lỗi in ra từ `[B]` (thường do thiếu file mô hình).
