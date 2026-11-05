"""Process B – suy luận: phát hiện -> chất lượng -> liveness -> embedding -> bỏ phiếu -> ghi SQLite/outbox.

Chạy trong process riêng (GIL riêng) để luồng camera và kiosk ở Process A không bị chặn.
"""
import csv
import os
import queue
import time
import traceback
from collections import deque
from datetime import datetime

import cv2

from . import config as config_mod
from .db import DB
from .decision import Decider, Vote
from .shm import FrameBuffer
from .vision import Gallery, Liveness, check_quality, make_detector, make_embedder

DEBUG = False


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        md = cfg["paths"]["models"]
        self.det = make_detector(cfg, md)
        lv = cfg["liveness"]
        self.live = Liveness(md, lv["models"], lv.get("crop_scale"), lv.get("threads", 2)) if lv["enabled"] else None
        name = cfg["embedder"]["name"]
        if name == "sface" and self.det.name != "yunet":
            raise ValueError("embedder sface cần detector yunet")
        self.emb = make_embedder(name, md)
        self.thr = cfg["embedder"]["match_threshold"][name]
        self.margin = cfg["embedder"]["margin"]
        self.E = cfg["embedder"]["embed_frames_per_track"][name]
        self.db = DB(cfg["paths"]["db"])
        self.reload_gallery()
        self.decider = Decider(cfg["decision"], self.E)

    def reload_gallery(self):
        self.gallery = Gallery(self.emb, self.db.gallery_rows(self.emb.model_version))
        self.names = self.db.names()
        self.gallery_n = len(self.gallery)

    def vote(self, frame, face, tr):
        ok, hint = check_quality(frame, face, self.cfg["quality"])
        if not ok:
            return Vote("skip", hint=hint)
        ls = self.live.score(frame, face) if self.live else 1.0
        if ls < self.cfg["liveness"]["threshold"]:
            return Vote("spoof", live=ls)
        if not self.decider.should_embed(tr):
            return Vote("live", live=ls)
        vec = self.emb.embed(frame, face)
        eid, d = self.gallery.match(vec, self.thr, self.margin)
        return Vote("id" if eid is not None else "unknown", live=ls, emp_id=eid, dist=d)

    def suspected(self, frame, face):
        try:
            vec = self.emb.embed(frame, face)
            eid, _ = self.gallery.match(vec, self.thr, self.margin)
            return eid
        except Exception:  # noqa: BLE001
            return None

    def handle(self, outcome, info, frame, face, tr):
        dev = self.cfg["device"]["id"]
        dc = self.cfg["decision"]
        res = {"state": outcome}
        if outcome == "attend":
            eid = info["emp_id"]
            res["name"] = self.names.get(eid, f"#{eid}")
            att, already = self.db.check_attendance(eid, dc["day_start"], dc["min_gap_min"])
            if att is None:
                res.update(state="already", already_at=already)
            else:
                self.db.record_attendance(dev, eid, att, info["live"], info["dist"], self.emb.model_version, info["votes"])
                res.update(att_type=att, time=datetime.now().strftime("%H:%M:%S"))
        elif outcome == "spoof":
            sid = self.suspected(frame, face) if len(self.gallery) else None
            snap = os.path.join(self.cfg["paths"]["snapshots"], datetime.now().strftime("spoof_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(snap, frame)
            self.db.record_spoof(dev, info["live"], sid, tr.attempts, snap)
            res["suspected"] = self.names.get(sid) if sid else None
        return res


def run(cfg_path, fb_handle, result_q, heartbeat, stop_evt, session="", ready_evt=None):
    import signal
    global DEBUG
    DEBUG = os.environ.get("ATT_DEBUG") == "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C do Process A xử lý, B tắt qua stop_evt
    cfg = config_mod.load(cfg_path)
    log_path = os.path.join(cfg["paths"]["logs"], "decisions.csv")
    new_log = not os.path.exists(log_path)
    flog = open(log_path, "a", newline="", encoding="utf-8")
    wlog = csv.writer(flog)
    if new_log:
        wlog.writerow(["time", "session", "outcome", "name", "votes", "n_valid", "liveness", "distance",
                       "decision_latency_s", "attempt", "inference_per_s", "embedder"])
    try:
        P = Pipeline(cfg)
    except Exception as e:  # noqa: BLE001
        result_q.put({"state": "error", "hint": f"Lỗi nạp mô hình: {e}"})
        traceback.print_exc()
        return
    fb = FrameBuffer.attach(fb_handle)
    if ready_evt is not None:
        ready_evt.set()
    D = P.decider
    show_s = cfg["decision"]["result_show_s"]
    last_seq, times, t_reload = -1, deque(maxlen=30), time.monotonic()
    stamps = deque(maxlen=60)  # thời điểm xử lý xong từng khung -> tốc độ thực tế
    print(f"[B] sẵn sàng: detector={P.det.name}, embedder={P.emb.name}, gallery={len(P.gallery)} embedding, E={P.E}",
          flush=True)
    while not stop_evt.is_set():
        heartbeat.value = time.time()
        if time.monotonic() - t_reload > 10:  # nạp lại danh sách nếu đã đăng ký thêm
            t_reload = time.monotonic()
            if len(P.db.gallery_rows(P.emb.model_version)) != P.gallery_n:
                P.reload_gallery()
        frame, seq = fb.read(last_seq)
        if frame is None:
            time.sleep(0.003)
            continue
        last_seq = seq
        t0 = time.perf_counter()
        faces = P.det.detect(frame)
        now = time.monotonic()
        face = max(faces, key=lambda f: f.area) if faces else None
        tr = D.observe(face, now)
        msg = {"seq": seq, "bbox": face.bbox if face else None, "state": "idle", "progress": 0.0, "hint": ""}
        try:
            if tr is not None and face is not None and not tr.decided:
                v = P.vote(frame, face, tr)
                D.add(tr, v)
                if DEBUG:
                    print(f"[dbg] {v.kind:7} live={'' if v.live is None else f'{v.live:.2f}'} "
                          f"sim/dist={'' if v.dist is None else f'{v.dist:.3f}'} mặt={int(face.bbox[2])}px "
                          f"{v.hint}", flush=True)
                out = D.evaluate(tr, now)
                if out and out[0] == "timeout":
                    wlog.writerow([datetime.now().isoformat(timespec="seconds"), session, "timeout", "", "",
                                   out[1].get("nv", ""), "", "", round(now - tr.t_start, 2), tr.attempts, "", P.emb.name])
                    flog.flush()
                    D.restart(tr, now)
                    msg.update(state="tracking", hint="Chưa xác nhận được, giữ yên khuôn mặt")
                elif out:
                    outcome, info = out
                    tr.decided = True
                    tr.result = P.handle(outcome, info, frame, face, tr)
                    tr.result_until = now + show_s
                    wlog.writerow([datetime.now().isoformat(timespec="seconds"), session, tr.result["state"],
                                   tr.result.get("name") or tr.result.get("suspected") or "", info.get("votes", ""),
                                   info.get("nv", ""), round(info.get("live", 0) or 0, 4),
                                   round(info["dist"], 4) if info.get("dist") is not None else "",
                                   round(now - tr.t_start, 2), tr.attempts,
                                   round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2) if len(stamps) > 5 else "", P.emb.name])
                    flog.flush()
                else:
                    msg.update(state="tracking", progress=D.progress(tr, now), hint=v.hint)
            if tr is not None and tr.decided:
                if now < tr.result_until:
                    msg.update(tr.result)
                else:
                    msg.update(state="next", hint="Mời người tiếp theo")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg.update(state="error", hint=str(e))
        times.append(time.perf_counter() - t0)
        stamps.append(time.monotonic())
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 5 and stamps[-1] > stamps[0] else 0.0
        msg["stats"] = {"inference_per_s": round(rate, 1), "capacity_per_s": round(len(times) / sum(times), 1),
                        "lat_ms": round(1000 * times[-1]),
                        "pending": P.db.outbox_pending(), "gallery": len(P.gallery)}
        try:
            result_q.put_nowait(msg)
        except queue.Full:
            try:
                result_q.get_nowait()
                result_q.put_nowait(msg)
            except (queue.Empty, queue.Full):
                pass
    flog.close()
    fb.close()