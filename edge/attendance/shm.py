"""Bộ đệm khung hình dùng chung giữa Process A (camera) và Process B (suy luận).

Hai slot: writer luôn ghi vào slot KHÔNG phải slot hiện hành rồi mới đổi chỉ số dưới lock,
reader sao chép slot hiện hành trong lúc giữ lock -> không bao giờ đọc khung đang ghi dở.
"""
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


class FrameBuffer:
    def __init__(self, shape, name=None, create=False, seq=None, cur=None, lock=None):
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(self.shape))
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=2 * self.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.buf = np.ndarray((2, *self.shape), dtype=np.uint8, buffer=self.shm.buf)
        self.seq = seq
        self.cur = cur
        self.lock = lock

    @classmethod
    def create(cls, shape, ctx=mp):
        return cls(shape, create=True, seq=ctx.Value("q", 0, lock=False), cur=ctx.Value("i", 0, lock=False),
                   lock=ctx.Lock())

    def handle(self):
        """Tham số truyền sang process khác."""
        return dict(shape=self.shape, name=self.shm.name, seq=self.seq, cur=self.cur, lock=self.lock)

    @classmethod
    def attach(cls, h):
        return cls(h["shape"], name=h["name"], seq=h["seq"], cur=h["cur"], lock=h["lock"])

    def write(self, frame):
        if frame.shape != self.shape:
            raise ValueError(f"frame {frame.shape} != {self.shape}")
        nxt = 1 - self.cur.value
        self.buf[nxt][...] = frame
        with self.lock:
            self.cur.value = nxt
            self.seq.value += 1

    def read(self, last_seq=-1):
        """Trả (frame_copy, seq) nếu có khung mới hơn last_seq, ngược lại (None, last_seq)."""
        with self.lock:
            s = self.seq.value
            if s == last_seq or s == 0:
                return None, last_seq
            frame = self.buf[self.cur.value].copy()
        return frame, s

    def close(self, unlink=False):
        self.buf = None
        self.shm.close()
        if unlink:
            self.shm.unlink()
