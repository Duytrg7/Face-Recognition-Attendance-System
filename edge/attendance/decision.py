"""Theo dõi khuôn mặt (track) và máy trạng thái quyết định theo bỏ phiếu k/N (thiết kế M1, mục 3.4)."""
import math
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class Vote:
    kind: str                 # skip | spoof | live | id | unknown
    live: float = None
    emp_id: int = None
    dist: float = None
    hint: str = ""


@dataclass
class Track:
    tid: int
    t_start: float
    t_seen: float
    bbox: tuple
    votes: deque = field(default_factory=lambda: deque(maxlen=15))
    n_embeds: int = 0
    decided: bool = False
    result: dict = None
    result_until: float = 0.0
    attempts: int = 1


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Decider:
    def __init__(self, dcfg, embed_frames):
        self.c = dcfg
        self.E = int(embed_frames)
        self.track = None
        self._next = 1
        # số phiếu danh tính cần có: E=0 (mọi khung) -> k_min; E>0 -> đa số trong E khung (vd 2/3)
        self.need_id = self.c["k_min"] if self.E == 0 else max(1, math.ceil(0.67 * self.E))

    # ------------------------------------------------------------ track
    def observe(self, face, now):
        """Cập nhật track theo mặt lớn nhất của khung. Trả track hiện tại (có thể None)."""
        tr = self.track
        if face is None:
            if tr and now - tr.t_seen > self.c["lost_s"]:
                self.track = None
            return self.track
        if tr is None or now - tr.t_seen > self.c["lost_s"] or iou(tr.bbox, face.bbox) < self.c["iou_track"]:
            tr = Track(self._next, now, now, face.bbox, deque(maxlen=self.c["window"]))
            self._next += 1
            self.track = tr
        tr.t_seen, tr.bbox = now, face.bbox
        return tr

    def should_embed(self, tr):
        return self.E == 0 or tr.n_embeds < self.E

    def add(self, tr, vote):
        tr.votes.append(vote)
        if vote.kind in ("id", "unknown"):
            tr.n_embeds += 1

    def restart(self, tr, now):
        """Hết T_out mà chưa quyết định: bắt đầu lượt mới trong cùng track."""
        tr.votes.clear()
        tr.n_embeds = 0
        tr.t_start = now
        tr.attempts += 1

    # ------------------------------------------------------------ quyết định
    def evaluate(self, tr, now):
        """Trả (kết_cục, chi_tiết) hoặc None. Kết cục: spoof | attend | unknown | timeout."""
        c = self.c
        valid = [v for v in tr.votes if v.kind != "skip"]
        nv = len(valid)
        spoof = [v for v in valid if v.kind == "spoof"]
        live = [v for v in valid if v.kind != "spoof"]
        elapsed = now - tr.t_start
        if (len(spoof) >= c["m_min"] and len(spoof) >= c["m_ratio"] * nv
                and elapsed >= c.get("t_min_spoof_s", 0.6)):
            return "spoof", {"n": len(spoof), "nv": nv, "live": min(v.live for v in spoof)}
        if elapsed >= c["t_min_s"] and len(live) >= c["k_min"] and len(live) >= c["k_ratio"] * nv:
            idv = [v for v in live if v.kind in ("id", "unknown")]
            n_emb = len(idv)
            if n_emb:
                cnt = Counter(v.emp_id for v in idv if v.kind == "id")
                if cnt:
                    best, nb = cnt.most_common(1)[0]
                    if nb >= self.need_id and nb >= c["k_ratio"] * n_emb:
                        ds = [v.dist for v in idv if v.kind == "id" and v.emp_id == best]
                        return "attend", {"emp_id": best, "votes": f"{nb}/{n_emb}", "nv": nv,
                                          "dist": sum(ds) / len(ds), "live": sum(v.live for v in live) / len(live)}
                nu = sum(1 for v in idv if v.kind == "unknown")
                if nu >= self.need_id and nu >= c["k_ratio"] * n_emb:
                    return "unknown", {"votes": f"{nu}/{n_emb}", "nv": nv,
                                       "live": sum(v.live for v in live) / len(live)}
        if elapsed > c["t_out_s"]:
            return "timeout", {"nv": nv}
        return None

    def progress(self, tr, now):
        live = [v for v in tr.votes if v.kind not in ("skip", "spoof")]
        a = min(1.0, (now - tr.t_start) / max(self.c["t_min_s"], 0.1))
        b = min(1.0, len(live) / max(self.c["k_min"], 1))
        return round(min(a, b), 2)
