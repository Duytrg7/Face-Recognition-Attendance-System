"""Process A – I/O: đọc camera vào shared memory, giám sát Process B (watchdog), hiển thị kiosk.

Chạy:  python -m attendance.app            (giao diện kiosk)
       python -m attendance.app --headless (không giao diện, in kết quả ra terminal – dùng qua SSH)
"""
import argparse
import multiprocessing as mp
import os
import queue
import signal
import threading
import time

from . import config as config_mod
from . import inference
from .camera import Source
from .shm import FrameBuffer


class EdgeApp:
    def __init__(self, cfg, session=""):
        self.cfg = cfg
        self.session = session
        self.ctx = mp.get_context("spawn")
        cam = cfg["camera"]
        self.fb = FrameBuffer.create((cam["height"], cam["width"], 3), self.ctx)
        self.q = self.ctx.Queue(maxsize=8)
        self.hb = self.ctx.Value("d", 0.0)
        self.stop_b = self.ctx.Event()
        self.stop = threading.Event()
        self.proc = None
        self.t_proc_start = 0.0
        self.latest = None          # khung mới nhất cho hiển thị
        self.cam_fps = 0.0
        self.last_msg = {"state": "loading", "hint": "Đang nạp mô hình…"}
        self.restarts = 0

    # ---------------------------------------------------------------- Process B
    def start_b(self):
        self.stop_b.clear()
        self.hb.value = 0.0
        self.proc = self.ctx.Process(target=inference.run, name="inference",
                                     args=(self.cfg["_path"], self.fb.handle(), self.q, self.hb, self.stop_b, self.session),
                                     daemon=True)
        self.proc.start()
        self.t_proc_start = time.time()

    def watchdog(self):
        w = self.cfg["watchdog"]
        while not self.stop.is_set():
            time.sleep(1)
            now = time.time()
            alive = self.proc is not None and self.proc.is_alive()
            stale = self.hb.value and now - self.hb.value > w["stale_s"]
            never = not self.hb.value and now - self.t_proc_start > w["startup_grace_s"]
            if not alive or stale or never:
                self.restarts += 1
                print(f"[A] watchdog: khởi động lại Process B (alive={alive}, stale={bool(stale)}, lần {self.restarts})",
                      flush=True)
                if self.proc is not None and self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(3)
                self.start_b()

    # ---------------------------------------------------------------- camera
    def grabber(self):
        src = Source(self.cfg["camera"])
        n, t0 = 0, time.monotonic()
        try:
            while not self.stop.is_set():
                ok, f = src.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                self.fb.write(f)
                self.latest = f
                n += 1
                dt = time.monotonic() - t0
                if dt >= 2:
                    self.cam_fps, n, t0 = n / dt, 0, time.monotonic()
        finally:
            src.release()

    def poll(self):
        """Lấy kết quả mới nhất từ Process B."""
        try:
            while True:
                self.last_msg = self.q.get_nowait()
        except queue.Empty:
            pass
        return self.last_msg

    def start(self):
        self.start_b()
        threading.Thread(target=self.grabber, daemon=True, name="grabber").start()
        threading.Thread(target=self.watchdog, daemon=True, name="watchdog").start()

    def shutdown(self):
        self.stop.set()
        self.stop_b.set()
        if self.proc is not None:
            self.proc.join(3)
            if self.proc.is_alive():
                self.proc.terminate()
        time.sleep(0.2)
        self.fb.close(unlink=True)


def run_headless(app, seconds=None):
    t_end = time.time() + seconds if seconds else None
    last_state = None
    while not app.stop.is_set() and (t_end is None or time.time() < t_end):
        m = app.poll()
        st = m.get("state")
        if st != last_state or st in ("attend", "already", "spoof", "unknown", "error"):
            if st != last_state:
                s = m.get("stats", {})
                extra = {k: m[k] for k in ("name", "att_type", "time", "already_at", "suspected", "hint") if m.get(k)}
                print(f"[{time.strftime('%H:%M:%S')}] {st:9} {extra}  cam={app.cam_fps:.1f}FPS "
                      f"infer={s.get('inference_per_s', '-')}/s (năng lực {s.get('capacity_per_s', '-')}/s) lat={s.get('lat_ms', '-')}ms "
                      f"pending={s.get('pending', '-')}", flush=True)
            last_state = st
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=None, help="chỉ chạy N giây (test)")
    ap.add_argument("--source", default=None, help="ghi đè camera.source (video/thư mục ảnh để test)")
    ap.add_argument("--session", default="", help="nhãn phiên thử ghi vào logs/decisions.csv, vd real:NV001, print, screen, video, unknown")
    args = ap.parse_args()
    cfg = config_mod.load(args.config)
    if args.source:
        cfg["camera"]["source"] = args.source
        if args.config is None:  # Process B đọc lại file cấu hình, nên ghi bản tạm
            import yaml
            tmp = os.path.join(cfg["paths"]["logs"], "_runtime_config.yaml")
            raw = yaml.safe_load(open(cfg["_path"], encoding="utf-8"))
            raw["camera"]["source"] = args.source
            yaml.safe_dump(raw, open(tmp, "w", encoding="utf-8"), allow_unicode=True)
            cfg["_path"] = tmp
    app = EdgeApp(cfg, args.session)
    signal.signal(signal.SIGTERM, lambda *_: app.stop.set())
    app.start()
    try:
        if args.headless:
            run_headless(app, args.seconds)
        else:
            from .kiosk import Kiosk
            Kiosk(app).run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
