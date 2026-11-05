import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Cfg(dict):
    """dict truy cập được bằng thuộc tính: cfg.camera.exposure"""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def load(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(ROOT, v)
    for k in ("db",):
        os.makedirs(os.path.dirname(cfg["paths"][k]), exist_ok=True)
    for k in ("snapshots", "enroll_images", "logs"):
        os.makedirs(cfg["paths"][k], exist_ok=True)
    cfg["_path"] = path
    return cfg
