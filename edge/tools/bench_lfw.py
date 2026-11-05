"""Benchmark độ chính xác xác minh khuôn mặt trên LFW (giao thức chuẩn 6000 cặp, 10 fold) cho 3 cấu hình:
  (a) HOG + dlib    – cấu hình ĐA1
  (b) YuNet + dlib
  (c) YuNet + SFace
Mục đích: chốt embedder cho M2 khi chưa có đủ tình nguyện viên.

Chạy được trên laptop (nhanh) hoặc Pi (chậm do dlib ~0,5s/ảnh: --folds 3 mất ~40 phút).
  python tools/bench_lfw.py                 # tự tải LFW (~180MB) vào data/lfw
  python tools/bench_lfw.py --folds 3       # chỉ dùng 3 fold (1800 cặp)
  python tools/bench_lfw.py --lfw-dir D:/lfw --pairs D:/pairs.txt
Kết quả: logs/bench_lfw.json (+ logs/bench_lfw_roc.png nếu có matplotlib). Có cache embedding, chạy lại sẽ tiếp tục.
"""
import argparse
import json
import os
import sys
import tarfile
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attendance.vision import DlibEmbedder, Face, SFaceEmbedder, YuNetDetector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_LFW = "https://ndownloader.figshare.com/files/5976018"     # lfw.tgz (bản gốc, 250x250)
URL_PAIRS = "https://ndownloader.figshare.com/files/5976006"   # pairs.txt


def download(url, dst):
    if os.path.exists(dst):
        return
    print(f"Tải {url} -> {dst}")
    tmp = dst + ".part"

    def hook(b, bs, total):
        if total > 0 and b % 200 == 0:
            print(f"\r  {min(100, b * bs * 100 // total)}%", end="", flush=True)
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, dst)
    print()


def get_lfw(args):
    if args.lfw_dir:
        return args.lfw_dir, args.pairs
    base = os.path.join(ROOT, "data", "lfw")
    os.makedirs(base, exist_ok=True)
    tgz, pairs = os.path.join(base, "lfw.tgz"), os.path.join(base, "pairs.txt")
    download(URL_PAIRS, pairs)
    if not os.path.isdir(os.path.join(base, "lfw")):
        download(URL_LFW, tgz)
        print("Giải nén…")
        with tarfile.open(tgz) as t:
            t.extractall(base, filter="data")
    return os.path.join(base, "lfw"), pairs


def read_pairs(path, folds):
    lines = open(path).read().strip().splitlines()
    nf, n = [int(x) for x in lines[0].split()]
    out, i = [], 1
    for f in range(nf):
        for k in range(2 * n):
            p = lines[i].split()
            i += 1
            if f >= folds:
                continue
            if len(p) == 3:
                out.append((f, 1, (p[0], int(p[1])), (p[0], int(p[2]))))
            else:
                out.append((f, 0, (p[0], int(p[1])), (p[2], int(p[3]))))
    return out


def img_path(root, name, idx):
    return os.path.join(root, name, f"{name}_{idx:04d}.jpg")


def central(faces, W, H):
    if not faces:
        return None
    return min(faces, key=lambda f: (f.bbox[0] + f.bbox[2] / 2 - W / 2) ** 2 + (f.bbox[1] + f.bbox[3] / 2 - H / 2) ** 2)


def fallback_face(W, H):  # mặt LFW nằm khoảng giữa ảnh 250x250
    s = 0.44 * W
    return Face(((W - s) / 2, (H - s) / 2 + 0.03 * H, s, s), 0.0)


def cv_accuracy(scores, labels, folds):
    """Chọn ngưỡng trên các fold còn lại, đo trên fold giữ ra (giao thức LFW)."""
    accs, ths = [], []
    uf = sorted(set(folds))
    for f in uf:
        tr, te = folds != f, folds == f
        cand = np.unique(scores[tr])
        best_t, best_a = cand[0], -1
        for t in cand[:: max(1, len(cand) // 2000)]:
            a = np.mean((scores[tr] >= t) == labels[tr])
            if a > best_a:
                best_a, best_t = a, t
        accs.append(np.mean((scores[te] >= best_t) == labels[te]))
        ths.append(best_t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.median(ths))


def roc(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    tpr = np.cumsum(l) / max(l.sum(), 1)
    fpr = np.cumsum(1 - l) / max((1 - l).sum(), 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = float((fpr[i] + 1 - tpr[i]) / 2)
    tar_at = {}
    for far in (1e-2, 1e-3):
        j = np.searchsorted(fpr, far, side="right") - 1
        tar_at[f"TAR@FAR={far:g}"] = float(tpr[j]) if j >= 0 else 0.0
    return fpr, tpr, auc, eer, tar_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw-dir")
    ap.add_argument("--pairs")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    lfw, pairs_txt = get_lfw(a)
    pairs = read_pairs(pairs_txt, a.folds)
    imgs = sorted({p for _, _, x, y in pairs for p in (x, y)})
    print(f"{len(pairs)} cặp, {len(imgs)} ảnh, {a.folds} fold")

    import dlib
    hog = dlib.get_frontal_face_detector()
    yunet = YuNetDetector(os.path.join(a.models, "face_detection_yunet_2023mar.onnx"), input_width=250, score=0.6)
    dl = DlibEmbedder(a.models)
    sf = SFaceEmbedder(a.models)
    cache = os.path.join(ROOT, "logs", f"bench_lfw_cache_{a.folds}.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    E = {"a": {}, "b": {}, "c": {}}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        E = z["E"].item()
    miss = {"hog": 0, "yunet": 0}
    tm = {"hog": [], "yunet": [], "dlib": [], "sface": []}
    t_save = time.time()
    for n, key in enumerate(imgs):
        if all(key in E[c] for c in "abc"):
            continue
        img = cv2.imread(img_path(lfw, *key))
        H, W = img.shape[:2]
        t = time.perf_counter()
        rects = hog(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        tm["hog"].append(time.perf_counter() - t)
        fh = central([Face((r.left(), r.top(), r.width(), r.height()), 1.0) for r in rects], W, H)
        if fh is None:
            miss["hog"] += 1
            fh = fallback_face(W, H)
        t = time.perf_counter()
        fy = central(yunet.detect(img), W, H)
        tm["yunet"].append(time.perf_counter() - t)
        t = time.perf_counter()
        E["a"][key] = dl.embed(img, fh)
        tm["dlib"].append(time.perf_counter() - t)
        if fy is None:
            miss["yunet"] += 1
            fb = fallback_face(W, H)
            E["b"][key] = dl.embed(img, fb)
            E["c"][key] = None
        else:
            E["b"][key] = dl.embed(img, fy)
            t = time.perf_counter()
            E["c"][key] = sf.embed(img, fy)
            tm["sface"].append(time.perf_counter() - t)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(imgs)} ảnh", flush=True)
        if time.time() - t_save > 120:
            np.savez(cache, E=np.array(E, dtype=object)); t_save = time.time()
    np.savez(cache, E=np.array(E, dtype=object))

    labels = np.array([p[1] for p in pairs])
    folds = np.array([p[0] for p in pairs])
    res = {"pairs": len(pairs), "images": len(imgs), "folds": a.folds, "detector_miss": miss,
           "ms_per_image": {k: round(1000 * float(np.mean(v)), 1) for k, v in tm.items() if v}, "configs": {}}
    curves = {}
    names = {"a": "HOG + dlib (ĐA1)", "b": "YuNet + dlib", "c": "YuNet + SFace"}
    for c in "abc":
        s = []
        for _, _, x, y in pairs:
            ex, ey = E[c][x], E[c][y]
            if ex is None or ey is None:
                s.append(-9.0)  # không phát hiện được mặt -> coi như khác người
            elif c == "c":
                s.append(float(np.dot(ex, ey)))
            else:
                s.append(-float(np.linalg.norm(ex - ey)))
        s = np.array(s)
        acc, sd, th = cv_accuracy(s, labels, folds)
        fpr, tpr, auc, eer, tar = roc(s, labels)
        curves[c] = (fpr, tpr)
        res["configs"][names[c]] = {"accuracy": round(acc, 4), "std": round(sd, 4),
                                    "threshold": round(th if c == "c" else -th, 4),
                                    "threshold_kind": "cosine >=" if c == "c" else "euclid <",
                                    "auc": round(auc, 4), "eer": round(eer, 4),
                                    **{k: round(v, 4) for k, v in tar.items()}}
    out = os.path.join(ROOT, "logs", "bench_lfw.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4.5), dpi=150)
        for c, col in zip("abc", ["#8a8a8a", "#eb6834", "#2a78d6"]):
            plt.plot(curves[c][0], curves[c][1], color=col, lw=2, label=f'{names[c]} (AUC {res["configs"][names[c]]["auc"]})')
        plt.xscale("log"); plt.xlim(1e-3, 1); plt.ylim(0.8, 1.0)
        plt.xlabel("FAR"); plt.ylabel("TAR"); plt.grid(alpha=0.3); plt.legend(loc="lower right", fontsize=8)
        plt.title(f"ROC trên LFW ({len(pairs)} cặp)")
        plt.tight_layout(); plt.savefig(os.path.join(ROOT, "logs", "bench_lfw_roc.png"))
    except ImportError:
        pass
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
