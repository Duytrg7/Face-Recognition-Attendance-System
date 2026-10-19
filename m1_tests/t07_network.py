"""T07 — Khối mạng Pi <-> server (chạy trên Pi, server_stub chạy trên laptop).

Kiểm tra: ping RTT, độ trễ HTTP, xác thực token, heartbeat + lệch giờ, idempotency event_uuid.
Chạy: python t07_network.py --server http://<IP_laptop>:8000 [--token m1-test-token] [--n 50]
"""
import argparse
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from common import cpu_temp, run, save_json, stats_ms

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--token", default="m1-test-token")
ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()
S = args.server.rstrip("/")
H = {"X-Device-Token": args.token}
DEV = "pi-m1"
R = {"server": S}


def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


host = urlparse(S).hostname
ping = run(f"ping -c 10 -q {host}")
m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)", ping)
R["ping_rtt_ms_min_avg_max"] = [float(x) for x in m.groups()] if m else ping
R["ping_loss"] = (re.search(r"(\d+)% packet loss", ping) or [None, "?"])[1]
R["wifi"] = run("iwconfig wlan0 2>/dev/null | grep -E 'ESSID|Bit Rate|Signal' | tr -s ' '")

# 1) GET /api/ping latency
lat = []
for _ in range(args.n):
    t = time.perf_counter(); requests.get(f"{S}/api/ping", timeout=5); lat.append(time.perf_counter() - t)
R["http_get_ping"] = stats_ms(lat)

# 2) auth: token sai phải bị 401
R["auth_wrong_token_status"] = requests.post(f"{S}/api/events", headers={"X-Device-Token": "wrong"},
                                             json={"event_uuid": str(uuid.uuid4()), "device_id": DEV,
                                                   "type": "attendance", "ts": iso_now()}, timeout=5).status_code

# 3) POST events latency (mỗi event một uuid mới)
lat = []
for i in range(args.n):
    ev = {"event_uuid": str(uuid.uuid4()), "device_id": DEV, "type": "attendance", "ts": iso_now(),
          "payload": {"employee_id": "TEST", "liveness_score": 0.9, "match_distance": 0.4, "seq": i}}
    t = time.perf_counter(); r = requests.post(f"{S}/api/events", headers=H, json=ev, timeout=5)
    lat.append(time.perf_counter() - t)
    assert r.status_code == 200, r.text
R["http_post_event"] = stats_ms(lat)

# 4) idempotency: gửi cùng 1 event 3 lần
ev = {"event_uuid": str(uuid.uuid4()), "device_id": DEV, "type": "attendance", "ts": iso_now(), "payload": {}}
c0 = requests.get(f"{S}/api/events/count", timeout=5).json()["events"]
statuses = [requests.post(f"{S}/api/events", headers=H, json=ev, timeout=5).json()["status"] for _ in range(3)]
c1 = requests.get(f"{S}/api/events/count", timeout=5).json()["events"]
R["idempotency"] = {"statuses": statuses, "count_increase": c1 - c0, "pass": statuses == ["created", "duplicate", "duplicate"] and c1 - c0 == 1}

# 5) heartbeat + clock skew
hb = requests.post(f"{S}/api/devices/heartbeat", headers=H, timeout=5,
                   json={"device_id": DEV, "device_time": iso_now(), "temp_c": cpu_temp(), "model_version": "m1-test"}).json()
R["heartbeat"] = hb

for k, v in R.items():
    print(f"{k:26}: {v}")
save_json("t07_network", R)
