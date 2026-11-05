"""Các khối thị giác: phát hiện mặt, kiểm tra chất lượng, liveness, embedding.

Lớp YuNet/SFace/Liveness dùng đúng các lời gọi OpenCV/ONNX Runtime đã chạy thành công trên Pi ở M1 (T03).
"""
import importlib.util
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    bbox: tuple          # x, y, w, h trên khung gốc
    score: float
    landmarks: np.ndarray = None   # 5x2: mắt phải, mắt trái, mũi, khoé miệng phải, trái
    row: np.ndarray = None         # hàng 15 giá trị kiểu YuNet trên khung gốc (cho SFace.alignCrop)

    @property
    def area(self):
        return self.bbox[2] * self.bbox[3]


# ============================================================ detector
class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path, input_width=320, score=0.8):
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 240), score, 0.3, 5000)
        self.input_width = input_width

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = self.input_width / w
        small = cv2.resize(frame, (self.input_width, int(round(h * s)))) if s != 1 else frame
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        out = []
        if faces is None:
            return out
        for r in faces:
            row = r.copy()
            row[:14] = row[:14] / s
            x, y, bw, bh = [float(v) for v in row[:4]]
            out.append(Face((x, y, bw, bh), float(row[14]), row[4:14].reshape(5, 2).copy(), row))
        return out


class HogDetector:
    """Dự phòng / chạy thử khi chưa có YuNet. Không có landmark."""
    name = "hog"

    def __init__(self, input_width=320, **_):
        import dlib
        self.det = dlib.get_frontal_face_detector()
        self.input_width = input_width

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = self.input_width / w
        small = cv2.resize(frame, (self.input_width, int(round(h * s))))
        rects = self.det(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), 0)
        return [Face((r.left() / s, r.top() / s, r.width() / s, r.height() / s), 1.0) for r in rects]


def make_detector(cfg, models_dir):
    d = cfg["detector"]
    if d["name"] == "yunet":
        return YuNetDetector(os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
                             d.get("input_width", 320), d.get("score_threshold", 0.8))
    return HogDetector(d.get("input_width", 320))


# ============================================================ quality
def check_quality(frame, face, q):
    """Trả (ok, lý_do). Lý do dùng để gợi ý trên kiosk."""
    H, W = frame.shape[:2]
    x, y, w, h = face.bbox
    m = q.get("margin_px", 10)
    if w < q["min_face_px"]:
        return False, "Lại gần camera hơn"
    if x < m or y < m or x + w > W - m or y + h > H - m:
        return False, "Đưa mặt vào giữa khung hình"
    crop = frame[int(max(y, 0)):int(min(y + h, H)), int(max(x, 0)):int(min(x + w, W))]
    if crop.size == 0:
        return False, "Đưa mặt vào giữa khung hình"
    g = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    br = float(g.mean())
    lo, hi = q["brightness"]
    if br < lo:
        return False, f"Thiếu sáng ({br:.0f}<{lo})"
    if br > hi:
        return False, f"Quá sáng ({br:.0f}>{hi})"
    if cv2.Laplacian(g, cv2.CV_64F).var() < q["min_sharpness"]:
        return False, "Giữ yên khuôn mặt"
    if face.landmarks is not None:
        re_, le_, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
        dr, dl = np.linalg.norm(nose - re_), np.linalg.norm(nose - le_)
        ratio = dr / max(dl, 1e-6)
        a, b = q["yaw_ratio"]
        if not a <= ratio <= b:
            return False, "Nhìn thẳng vào camera"
    return True, ""


# ============================================================ liveness
class Liveness:
    """MiniFASNet x2 (ONNX). Điểm real trung bình trong [0,1]."""

    def __init__(self, models_dir, models, crop_scale=None, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sessions = []
        for fname, native_scale in models:
            sess = ort.InferenceSession(os.path.join(models_dir, fname), so, providers=["CPUExecutionProvider"])
            self.sessions.append((sess, float(crop_scale or native_scale)))

    @staticmethod
    def crop(img, bbox, scale, out=80):
        src_h, src_w = img.shape[:2]
        x, y, bw, bh = bbox
        scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = x + bw / 2, y + bh / 2
        x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if x1 < 0:
            x2 -= x1; x1 = 0
        if y1 < 0:
            y2 -= y1; y1 = 0
        if x2 > src_w - 1:
            x1 -= x2 - src_w + 1; x2 = src_w - 1
        if y2 > src_h - 1:
            y1 -= y2 - src_h + 1; y2 = src_h - 1
        patch = img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
        return cv2.resize(patch, (out, out))

    def score(self, frame, face):
        prob = np.zeros(3, dtype=np.float32)
        for sess, s in self.sessions:
            x = self.crop(frame, face.bbox, s).transpose(2, 0, 1)[None].astype(np.float32)
            prob += sess.run(None, {"input": x})[0][0]
        return float(prob[1] / len(self.sessions))


# ============================================================ embedder
class Embedder:
    name = "base"
    model_version = "base"
    higher_is_closer = False

    def embed(self, frame, face):
        raise NotImplementedError

    def distance(self, a, b):
        raise NotImplementedError

    def is_match(self, d, thr):
        return d >= thr if self.higher_is_closer else d < thr


class SFaceEmbedder(Embedder):
    name = "sface"
    model_version = "sface-2021dec"
    higher_is_closer = True  # cosine similarity

    def __init__(self, models_dir):
        self.rec = cv2.FaceRecognizerSF.create(os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"), "")

    def embed(self, frame, face):
        if face.row is None:
            raise ValueError("SFace cần landmark của YuNet (detector: yunet)")
        aligned = self.rec.alignCrop(frame, face.row)
        f = self.rec.feature(aligned).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def distance(self, a, b):  # cosine (đã chuẩn hoá)
        return float(np.dot(a, b))


def _frm_models():
    spec = importlib.util.find_spec("face_recognition_models")
    if spec is None:
        raise ImportError("Thiếu face_recognition_models (pip install face_recognition_models)")
    base = os.path.join(list(spec.submodule_search_locations)[0], "models")
    return (os.path.join(base, "shape_predictor_5_face_landmarks.dat"),
            os.path.join(base, "dlib_face_recognition_resnet_model_v1.dat"))


class DlibEmbedder(Embedder):
    """Tương đương face_recognition.face_encodings(model='small') của ĐA1, gọi dlib trực tiếp."""
    name = "dlib"
    model_version = "dlib-resnet-v1"
    higher_is_closer = False  # euclid

    def __init__(self, models_dir=None):
        import dlib
        sp, rec = _frm_models()
        self.dlib = dlib
        self.sp = dlib.shape_predictor(sp)
        self.rec = dlib.face_recognition_model_v1(rec)

    def embed(self, frame, face):
        x, y, w, h = face.bbox
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect = self.dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = self.sp(rgb, rect)
        return np.array(self.rec.compute_face_descriptor(rgb, shape, 1), dtype=np.float32)

    def distance(self, a, b):
        return float(np.linalg.norm(a - b))


def make_embedder(name, models_dir):
    return SFaceEmbedder(models_dir) if name == "sface" else DlibEmbedder(models_dir)


# ============================================================ gallery
class Gallery:
    """Danh sách embedding đã đăng ký (cùng model_version với embedder đang chạy)."""

    def __init__(self, embedder, rows):
        self.emb = embedder
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.vecs = np.stack([r[1] for r in rows]) if rows else np.zeros((0, 128), np.float32)

    def __len__(self):
        return len(self.ids)

    def match(self, vec, thr, margin):
        """Trả (employee_id | None, khoảng cách tốt nhất)."""
        if len(self.ids) == 0:
            return None, None
        if self.emb.higher_is_closer:
            d = self.vecs @ vec
            order = np.argsort(-d)
        else:
            d = np.linalg.norm(self.vecs - vec, axis=1)
            order = np.argsort(d)
        best = int(order[0])
        bid, bd = int(self.ids[best]), float(d[best])
        # độ chênh với NGƯỜI khác gần nhất (bỏ qua các embedding cùng người)
        other = next((float(d[i]) for i in order[1:] if self.ids[i] != bid), None)
        if not self.emb.is_match(bd, thr):
            return None, bd
        if other is not None and abs(bd - other) < margin:
            return None, bd
        return bid, bd