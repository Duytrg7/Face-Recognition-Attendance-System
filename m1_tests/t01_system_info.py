"""T01 — Thông tin hệ thống Pi: OS, kernel, RAM, nhiệt độ, throttling, phiên bản thư viện."""
import importlib
import platform

from common import cpu_freq_mhz, cpu_temp, run, save_json, throttled


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({type(e).__name__})"


info = {
    "model": run("cat /proc/device-tree/model | tr -d '\\0'"),
    "os": run("grep PRETTY_NAME /etc/os-release | cut -d= -f2"),
    "kernel": platform.release(),
    "arch": platform.machine(),
    "python": platform.python_version(),
    "ram": run("free -h | awk '/Mem:/{print $2\" total, \"$7\" available\"}'"),
    "disk": run("df -h / | awk 'NR==2{print $2\" total, \"$4\" free\"}'"),
    "cpu_temp_c": cpu_temp(),
    "cpu_freq_mhz": cpu_freq_mhz(),
    "throttled": throttled(),
    "libs": {m: ver(m) for m in ["cv2", "numpy", "dlib", "face_recognition", "onnxruntime",
                                 "customtkinter", "requests"]},
    "opencv_has_yunet": hasattr(importlib.import_module("cv2"), "FaceDetectorYN"),
    "usb_devices": run("lsusb"),
    "video_devices": run("v4l2-ctl --list-devices"),
    "i2c_enabled": run("ls /dev/i2c-1 2>/dev/null || echo 'NO /dev/i2c-1'"),
    "rtc_devices": run("ls /dev/rtc* 2>/dev/null || echo 'NO /dev/rtc'"),
}
for k, v in info.items():
    print(f"{k:18}: {v}")
save_json("t01_system_info", info)
