"""Tổng hợp logs/decisions.csv theo nhãn phiên thử (--session khi chạy app).

Nhãn:  real:<MÃ hoặc tên>  – người đã đăng ký đứng thật   -> đúng nếu attend/already đúng người
       unknown              – người chưa đăng ký           -> đúng nếu unknown
       print | screen | video – tấn công                   -> đúng nếu spoof
Chỉ số (sơ bộ, trên 2 người – không dùng làm kết quả cuối):
  BPCER = lượt mặt thật bị báo spoof / lượt mặt thật
  APCER = lượt tấn công KHÔNG bị chặn (attend/already/unknown) / lượt tấn công, theo từng kiểu
  Nhận diện đúng = lượt mặt thật ra đúng người / lượt mặt thật có quyết định
  FAR (người lạ) = lượt unknown bị nhận thành người đã đăng ký / lượt unknown
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attendance import config as config_mod  # noqa: E402
from attendance.db import DB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = config_mod.load(a.config)
    path = a.log or os.path.join(cfg["paths"]["logs"], "decisions.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("session")]
    if not rows:
        sys.exit("Chưa có lượt thử nào được gắn --session")
    code2name = {c: n for _, c, n, *_ in DB(cfg["paths"]["db"]).employees()}
    groups = defaultdict(list)
    for r in rows:
        kind = r["session"].split(":")[0]
        groups[kind].append(r)
    out = {}
    for kind, rs in groups.items():
        dec = [r for r in rs if r["outcome"] != "timeout"]
        n, nt = len(dec), len(rs) - len(dec)
        cnt = defaultdict(int)
        for r in dec:
            cnt[r["outcome"]] += 1
        lat = [float(r["decision_latency_s"]) for r in dec if r["decision_latency_s"]]
        res = {"luot_quyet_dinh": n, "timeout": nt, "ket_cuc": dict(cnt),
               "do_tre_quyet_dinh_s": {"tb": round(st.mean(lat), 2), "max": round(max(lat), 2)} if lat else None}
        if kind == "real":
            correct = 0
            for r in dec:
                exp = r["session"].split(":", 1)[1] if ":" in r["session"] else ""
                exp_name = code2name.get(exp, exp)
                if r["outcome"] in ("attend", "already") and (not exp_name or r["name"] == exp_name):
                    correct += 1
            res["nhan_dien_dung"] = f"{correct}/{n}" + (f" ({100 * correct / n:.1f}%)" if n else "")
            res["BPCER"] = f'{cnt["spoof"]}/{n}' + (f' ({100 * cnt["spoof"] / n:.1f}%)' if n else "")
        elif kind == "unknown":
            fa = cnt["attend"] + cnt["already"]
            res["FAR_nguoi_la"] = f"{fa}/{n}" + (f" ({100 * fa / n:.1f}%)" if n else "")
        else:
            passed = n - cnt["spoof"]
            res["APCER"] = f"{passed}/{n}" + (f" ({100 * passed / n:.1f}%)" if n else "")
        out[kind] = res
    rates = [float(r["inference_per_s"]) for r in rows if r.get("inference_per_s")]
    if rates:
        out["toc_do_suy_luan_thuc_te_per_s"] = {"tb": round(st.mean(rates), 2), "min": round(min(rates), 2)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dst = os.path.join(cfg["paths"]["logs"], "summary_m2.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {dst}")


if __name__ == "__main__":
    main()
