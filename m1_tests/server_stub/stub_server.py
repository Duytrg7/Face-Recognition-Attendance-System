"""Server stub cho M1 — chạy trên LAPTOP. Chỉ để chứng minh khối mạng Pi <-> server hoạt động
và kiểm tra idempotency theo event_uuid. Không phải server thật của đồ án.

Cài:  pip install fastapi uvicorn
Chạy: DEVICE_TOKEN=m1-test-token uvicorn stub_server:app --host 0.0.0.0 --port 8000
"""
import os
import sqlite3
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

TOKEN = os.environ.get("DEVICE_TOKEN", "m1-test-token")
db = sqlite3.connect("stub.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS events (event_uuid TEXT PRIMARY KEY, device_id TEXT, "
           "type TEXT, ts TEXT, payload TEXT, received_at TEXT)")
db.execute("CREATE TABLE IF NOT EXISTS heartbeats (device_id TEXT, device_time TEXT, server_time TEXT, "
           "payload TEXT)")
db.commit()
app = FastAPI(title="M1 stub")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def auth(token):
    if token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid device token")


class Event(BaseModel):
    event_uuid: str
    device_id: str
    type: str          # attendance | spoof
    ts: str            # ISO-8601 UTC, giờ thiết bị
    payload: dict = {}


class Heartbeat(BaseModel):
    device_id: str
    device_time: str
    fps: float | None = None
    temp_c: float | None = None
    model_version: str | None = None


@app.get("/api/ping")
def ping():
    return {"ok": True, "server_time": now()}


@app.post("/api/events")
def post_event(ev: Event, x_device_token: str = Header(default="")):
    auth(x_device_token)
    try:
        db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                   (ev.event_uuid, ev.device_id, ev.type, ev.ts, str(ev.payload), now()))
        db.commit()
        return {"status": "created", "event_uuid": ev.event_uuid}
    except sqlite3.IntegrityError:  # gửi lại -> không tạo bản ghi đôi
        return {"status": "duplicate", "event_uuid": ev.event_uuid}


@app.post("/api/devices/heartbeat")
def heartbeat(hb: Heartbeat, x_device_token: str = Header(default="")):
    auth(x_device_token)
    st = now()
    db.execute("INSERT INTO heartbeats VALUES (?,?,?,?)", (hb.device_id, hb.device_time, st, hb.model_dump_json()))
    db.commit()
    skew = (datetime.fromisoformat(hb.device_time) - datetime.fromisoformat(st)).total_seconds()
    return {"ok": True, "server_time": st, "clock_skew_s": round(skew, 3)}


@app.get("/api/events/count")
def count():
    return {"events": db.execute("SELECT COUNT(*) FROM events").fetchone()[0]}
