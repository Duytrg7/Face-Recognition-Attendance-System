"""T03 — Khối RTC DS3231 (I2C).
Chuẩn bị (1 lần, xem README mục RTC): bật I2C, thêm `dtoverlay=i2c-rtc,ds3231` vào
/boot/firmware/config.txt, reboot.

Chạy:  sudo venv/bin/python t03_rtc.py            -> kiểm tra bus, đọc giờ RTC, so với giờ hệ thống
       sudo venv/bin/python t03_rtc.py --write    -> ghi giờ hệ thống (đã NTP) vào RTC
       sudo venv/bin/python t03_rtc.py --offline  -> sau khi tắt mạng + rút điện, kiểm tra giờ còn đúng
"""
import argparse
import time
from datetime import datetime, timezone

from common import run, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
ap.add_argument("--offline", action="store_true")
args = ap.parse_args()

r = {"mode": "write" if args.write else "offline-check" if args.offline else "check"}
r["i2cdetect"] = run("i2cdetect -y 1")
# 0x68 = DS3231. Khi driver rtc đã nhận chip, ô 0x68 hiện 'UU'.
r["chip_at_0x68"] = ("68" in r["i2cdetect"]) or ("UU" in r["i2cdetect"])
r["rtc_dev"] = run("ls -l /dev/rtc0 2>&1")
r["rtc_name"] = run("cat /sys/class/rtc/rtc0/name 2>&1")
r["timedatectl"] = run("timedatectl")
r["ntp_synced"] = "System clock synchronized: yes" in r["timedatectl"]
r["network_up"] = run("ip route | grep -c default") not in ("", "0")

if args.write:
    r["write_out"] = run("hwclock -w 2>&1")

rtc_raw = run("hwclock -r 2>&1")
sys_now = datetime.now(timezone.utc).astimezone()
r["rtc_read"] = rtc_raw
r["system_time"] = sys_now.isoformat(timespec="seconds")
try:
    rtc_dt = datetime.fromisoformat(rtc_raw.strip().replace(" ", "T", 1).split(".")[0])
    if rtc_dt.tzinfo is None:
        rtc_dt = rtc_dt.astimezone()
    r["rtc_minus_system_s"] = round((rtc_dt - sys_now).total_seconds(), 2)
except Exception as e:  # noqa: BLE001
    r["parse_error"] = str(e)

for k in ["chip_at_0x68", "rtc_name", "ntp_synced", "network_up", "rtc_read", "system_time",
          "rtc_minus_system_s"]:
    print(f"{k:20}: {r.get(k)}")
print("\nĐạt nếu: chip_at_0x68=True, rtc_name chứa ds3231/ds1307, |rtc_minus_system_s| <= 2.")
if args.offline:
    print("Ở chế độ --offline: network_up=False và system_time phải đúng giờ thật (so với điện thoại).")
save_json(f"t03_rtc_{r['mode']}_{int(time.time())}", r)
