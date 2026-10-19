# Bộ test M1 — chạy thử từng khối trên Raspberry Pi 4

Mỗi script ghi kết quả vào `logs/` (JSON/CSV + ảnh mẫu). Chạy xong thì **nén cả thư mục `logs/` gửi lại** để điền vào báo cáo M1.

| Script | Khối | Thời gian | Chạy ở đâu |
|---|---|---|---|
| `t01_system_info.py` | Hệ thống Pi, thư viện, thiết bị USB/I²C | 10 s | Pi |
| `t02_camera.py` | Camera C270: FPS, khoá exposure/WB | ~30 s | Pi |
| `t03_rtc.py` | RTC DS3231 | 5 phút (có rút điện) | Pi (sudo) |
| `t04_bench_models.py` | Độ trễ HOG / YuNet / dlib / SFace / liveness | ~3 phút | Pi |
| `t05_detect_distance.py` | Tỉ lệ phát hiện mặt theo 0.5 / 1 / 1.5 m | ~2 phút | Pi |
| `t06_stress.py` | Pipeline đầy đủ chạy liên tục: nhiệt độ, throttling, tốc độ | 15 phút × 2 | Pi |
| `server_stub/stub_server.py` + `t07_network.py` | Mạng Pi ↔ server, token, idempotency | 1 phút | Laptop + Pi |

## 0. Chuẩn bị (một lần)

```bash
# copy thư mục m1_tests lên Pi, rồi:
cd m1_tests
source ~/face_rec/bin/activate          # venv của ĐA1 (đổi đường dẫn nếu khác)
pip install onnxruntime requests        # OpenCV >= 4.8 để có YuNet/SFace
python -c "import cv2; print(cv2.__version__, hasattr(cv2,'FaceDetectorYN'))"
#   -> nếu False: pip install "opencv-python>=4.8"   (hoặc opencv-contrib-python)
sudo apt install -y v4l-utils i2c-tools
bash download_models.sh                 # tải YuNet + SFace
```

- `models/` đã có sẵn 2 file liveness **MiniFASNet đã chuyển sang ONNX** (từ Silent-Face-Anti-Spoofing, Apache-2.0; đã kiểm tra khớp PyTorch, sai số ~1e-7). Script chuyển đổi: `tools_convert_minifasnet.py`.
- Nếu `import face_recognition` báo thiếu `face_recognition_models` dù đã cài → `pip install "setuptools<81"` (setuptools mới bỏ `pkg_resources`).
- **Tắt GUI ĐA1** trước khi test (camera chỉ mở được bởi một chương trình).

## 1. T01 — thông tin hệ thống
```bash
python t01_system_info.py
```

## 2. T02 — camera
```bash
python t02_camera.py --seconds 10
```
Trong 10 s của mỗi chế độ, **đi qua lại trước camera hoặc che/bật đèn** để thấy chế độ AUTO dao động độ sáng.
Exposure của C270 tính theo đơn vị 100 µs: giá trị > 333 (33 ms) sẽ làm FPS tụt dưới 30. Nếu ảnh LOCKED quá tối/sáng thì chạy lại với `--exposure 200` / `--exposure 400`, chọn giá trị đẹp nhất và **ghi lại giá trị đó** — sẽ dùng cố định cho thu dữ liệu và triển khai.

## 3. T03 — RTC DS3231

Nối dây (module DS3231 → header Pi): **VCC → pin 1 (3.3V), SDA → pin 3 (GPIO2), SCL → pin 5 (GPIO3), GND → pin 6**.
> Module ZS-042 có mạch sạc cho pin LIR2032. Nếu dùng **CR2032 (không sạc)** thì nên gỡ điện trở/diode sạc trên module.

Cấu hình (một lần):
```bash
sudo raspi-config nonint do_i2c 0
echo "dtoverlay=i2c-rtc,ds3231" | sudo tee -a /boot/firmware/config.txt   # Bookworm; bản cũ: /boot/config.txt
sudo apt -y remove fake-hwclock && sudo update-rc.d -f fake-hwclock remove 2>/dev/null; sudo systemctl disable fake-hwclock 2>/dev/null
sudo reboot
```
Kiểm tra:
```bash
PY=$(which python)
sudo $PY t03_rtc.py            # 0x68 phải hiện 'UU'
sudo $PY t03_rtc.py --write    # khi Pi đang có mạng (NTP đã đồng bộ) -> ghi giờ vào RTC
# Tắt Wi-Fi + rút điện Pi ~3 phút, cắm lại KHÔNG có mạng:
sudo rfkill block wifi   # (trước khi rút điện, hoặc tắt router/hotspot)
sudo $PY t03_rtc.py --offline  # system_time phải đúng giờ thật -> chụp màn hình kèm đồng hồ điện thoại
sudo rfkill unblock wifi
```

## 4. T04 — độ trễ từng mô hình
```bash
python t04_bench_models.py --frames 60
```
Đứng trước camera ~1 m, nhìn thẳng, đủ sáng. Kết quả chính ở `pipeline_estimate`.

## 5. T05 — phát hiện mặt theo khoảng cách
```bash
python t05_detect_distance.py
```
Đo khoảng cách bằng thước (0.5 / 1.0 / 1.5 m), làm theo hướng dẫn trên màn hình.

## 6. T06 — chạy liên tục (nhiệt độ, throttling)
Hiển thị qua **HDMI capture, tắt VNC**:
```bash
python t06_stress.py --minutes 15 --detector yunet --embedder dlib --tag vnc_off
```
Sau đó bật VNC (có client đang kết nối) và chạy lại:
```bash
python t06_stress.py --minutes 15 --detector yunet --embedder dlib --tag vnc_on
```
Cần có mặt người (hoặc ảnh/điện thoại hiện khuôn mặt) trước camera suốt thời gian chạy. Nếu còn thời gian, chạy thêm `--embedder sface`.

## 7. T07 — mạng Pi ↔ server
Trên **laptop**:
```bash
cd server_stub
pip install fastapi uvicorn
DEVICE_TOKEN=m1-test-token uvicorn stub_server:app --host 0.0.0.0 --port 8000
#   Windows PowerShell:  $env:DEVICE_TOKEN="m1-test-token"; uvicorn stub_server:app --host 0.0.0.0 --port 8000
```
(Cho phép port 8000 qua firewall của laptop.) Trên **Pi**:
```bash
python t07_network.py --server http://<IP_laptop>:8000
```

## Gửi lại
- Thư mục `logs/` (nén zip).
- Ảnh chụp: phần cứng đã đấu nối (Pi + C270 + RTC + capture card), màn hình kiosk qua HDMI capture, kết quả `t03 --offline` cạnh đồng hồ điện thoại.
- Giá trị exposure đã chọn ở T02.
