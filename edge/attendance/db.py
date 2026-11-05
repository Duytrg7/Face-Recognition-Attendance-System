"""CSDL cục bộ trên Pi (SQLite, WAL). Bản ghi chấm công và outbox ghi trong CÙNG một transaction."""
import json
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
  employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT UNIQUE NOT NULL,
  full_name TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id INTEGER NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
  model_version TEXT NOT NULL,
  vector BLOB NOT NULL,
  source TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model_version);
CREATE TABLE IF NOT EXISTS local_attendance (
  event_uuid TEXT PRIMARY KEY,
  employee_id INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  type_provisional TEXT NOT NULL,
  liveness_score REAL,
  match_distance REAL,
  time_synced INTEGER,
  boot_id TEXT,
  mono_s REAL
);
CREATE INDEX IF NOT EXISTS idx_att_emp_ts ON local_attendance(employee_id, ts_utc);
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_uuid TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  snapshot_path TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  next_retry_at TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_state (key TEXT PRIMARY KEY, value TEXT);
"""


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def boot_id():
    try:
        return open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return "unknown"


_sync_cache = {"t": 0.0, "v": None}


def time_synced():
    """timedatectl NTPSynchronized, cache 60s."""
    if time.monotonic() - _sync_cache["t"] > 60:
        try:
            out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                                 capture_output=True, text=True, timeout=3).stdout.strip()
            _sync_cache["v"] = out == "yes" if out else None
        except Exception:  # noqa: BLE001
            _sync_cache["v"] = None
        _sync_cache["t"] = time.monotonic()
    return _sync_cache["v"]


class DB:
    def __init__(self, path):
        self.con = sqlite3.connect(path, timeout=3, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA busy_timeout=3000")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)
        self.con.commit()

    # ---------------------------------------------------------- nhân viên
    def add_employee(self, code, full_name):
        cur = self.con.execute("INSERT OR IGNORE INTO employees(code, full_name, created_at) VALUES (?,?,?)",
                               (code, full_name, iso(now_utc())))
        self.con.execute("UPDATE employees SET full_name=? WHERE code=?", (full_name, code))
        self.con.commit()
        return self.con.execute("SELECT employee_id FROM employees WHERE code=?", (code,)).fetchone()[0]

    def add_embedding(self, employee_id, model_version, vec, source):
        self.con.execute("INSERT INTO embeddings(employee_id, model_version, vector, source, created_at) VALUES (?,?,?,?,?)",
                         (employee_id, model_version, np.asarray(vec, np.float32).tobytes(), source, iso(now_utc())))

    def commit(self):
        self.con.commit()

    def delete_employee(self, code):
        self.con.execute("DELETE FROM embeddings WHERE employee_id IN (SELECT employee_id FROM employees WHERE code=?)", (code,))
        n = self.con.execute("DELETE FROM employees WHERE code=?", (code,)).rowcount
        self.con.commit()
        return n

    def employees(self):
        return self.con.execute(
            "SELECT e.employee_id, e.code, e.full_name, "
            "(SELECT COUNT(*) FROM embeddings b WHERE b.employee_id=e.employee_id AND b.model_version='sface-2021dec'),"
            "(SELECT COUNT(*) FROM embeddings b WHERE b.employee_id=e.employee_id AND b.model_version='dlib-resnet-v1') "
            "FROM employees e ORDER BY e.code").fetchall()

    def names(self):
        return {r[0]: r[1] for r in self.con.execute("SELECT employee_id, full_name FROM employees")}

    def gallery_rows(self, model_version):
        rows = self.con.execute("SELECT employee_id, vector FROM embeddings WHERE model_version=?", (model_version,))
        return [(eid, np.frombuffer(v, dtype=np.float32).copy()) for eid, v in rows]

    # ---------------------------------------------------------- chấm công
    def _day_start_utc(self, day_start="04:00"):
        loc = datetime.now().astimezone()
        hh, mm = [int(x) for x in day_start.split(":")]
        start = loc.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if loc < start:
            start -= timedelta(days=1)
        return start.astimezone(timezone.utc)

    def last_today(self, employee_id, day_start):
        return self.con.execute(
            "SELECT type_provisional, ts_utc FROM local_attendance WHERE employee_id=? AND ts_utc>=? "
            "ORDER BY ts_utc DESC LIMIT 1", (employee_id, iso(self._day_start_utc(day_start)))).fetchone()

    def check_attendance(self, employee_id, day_start, min_gap_min):
        """Trả ('in'|'out', None) nếu được chấm; (None, 'hh:mm') nếu đã chấm trong min_gap."""
        last = self.last_today(employee_id, day_start)
        if last:
            t_last = datetime.fromisoformat(last[1].replace("Z", "+00:00"))
            if now_utc() - t_last < timedelta(minutes=min_gap_min):
                return None, t_last.astimezone().strftime("%H:%M")
            return ("out" if last[0] == "in" else "in"), None
        return "in", None

    def record_attendance(self, device_id, employee_id, att_type, liveness, distance, model_version, votes):
        ev, ts = str(uuid.uuid4()), now_utc()
        synced = time_synced()
        payload = {"employee_id": employee_id, "type_provisional": att_type, "liveness_score": round(liveness, 4),
                   "match_distance": round(distance, 4) if distance is not None else None,
                   "model_version": model_version, "votes": votes}
        with self.con:  # một transaction
            self.con.execute("INSERT INTO local_attendance VALUES (?,?,?,?,?,?,?,?,?)",
                             (ev, employee_id, iso(ts), att_type, liveness, distance,
                              None if synced is None else int(synced), boot_id(), time.monotonic()))
            self.con.execute("INSERT INTO outbox(event_uuid, kind, payload, created_at) VALUES (?,?,?,?)",
                             (ev, "attendance", json.dumps({"event_uuid": ev, "device_id": device_id, "type": "attendance",
                                                            "ts": iso(ts), "time_synced": synced, "payload": payload}),
                              iso(ts)))
        return ev, ts

    def record_spoof(self, device_id, liveness, suspected_id, attempts, snapshot_path):
        ev, ts = str(uuid.uuid4()), now_utc()
        payload = {"liveness_score": round(liveness, 4), "suspected_employee_id": suspected_id,
                   "attempts": attempts, "has_snapshot": bool(snapshot_path)}
        with self.con:
            self.con.execute("INSERT INTO outbox(event_uuid, kind, payload, snapshot_path, created_at) VALUES (?,?,?,?,?)",
                             (ev, "spoof", json.dumps({"event_uuid": ev, "device_id": device_id, "type": "spoof",
                                                       "ts": iso(ts), "time_synced": time_synced(), "payload": payload}),
                              snapshot_path, iso(ts)))
        return ev, ts

    def outbox_pending(self):
        return self.con.execute("SELECT COUNT(*) FROM outbox WHERE status='pending'").fetchone()[0]

    def recent(self, limit=10):
        return self.con.execute(
            "SELECT a.ts_utc, e.full_name, a.type_provisional FROM local_attendance a "
            "JOIN employees e USING(employee_id) ORDER BY a.ts_utc DESC LIMIT ?", (limit,)).fetchall()
