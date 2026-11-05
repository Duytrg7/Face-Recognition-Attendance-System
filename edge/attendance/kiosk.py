"""Giao diện kiosk (CustomTkinter) – chạy trong main thread của Process A."""
import time
from datetime import datetime

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from .db import DB

STYLE = {  # state: (nền panel, màu khung mặt BGR, tiêu đề)
    "loading": ("#1d2633", (160, 160, 160), "Đang khởi động…"),
    "idle": ("#1d2633", (160, 160, 160), "Vui lòng đứng trước camera"),
    "tracking": ("#1d2633", (214, 120, 42), "Đang xác nhận…"),
    "attend": ("#12351f", (122, 175, 27), "Xin chào"),
    "already": ("#2b2f12", (0, 200, 230), "Đã chấm công"),
    "unknown": ("#3a2a12", (0, 160, 240), "Chưa đăng ký"),
    "spoof": ("#4a1414", (60, 76, 231), "⚠ KHÔNG HỢP LỆ"),
    "next": ("#1d2633", (160, 160, 160), "Mời người tiếp theo"),
    "error": ("#4a1414", (60, 76, 231), "Lỗi hệ thống"),
}
TYPE_VN = {"in": "VÀO", "out": "RA"}


class Kiosk:
    def __init__(self, app):
        self.app = app
        cfg = app.cfg
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title(cfg["kiosk"]["title"])
        self.root.geometry("1040x600")
        if cfg["kiosk"]["fullscreen"]:
            self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.db = DB(cfg["paths"]["db"])
        self.period = int(1000 / cfg["kiosk"]["display_fps"])

        top = ctk.CTkFrame(self.root, height=34, fg_color="#0d0f13", corner_radius=0)
        top.pack(fill="x")
        self.lbl_status = ctk.CTkLabel(top, text="", font=("Arial", 13), text_color="#cfd6e0")
        self.lbl_status.pack(side="left", padx=12)
        self.lbl_clock = ctk.CTkLabel(top, text="", font=("Arial", 13), text_color="#cfd6e0")
        self.lbl_clock.pack(side="right", padx=12)

        body = ctk.CTkFrame(self.root, fg_color="#14171c", corner_radius=0)
        body.pack(fill="both", expand=True)
        self.video = ctk.CTkLabel(body, text="", width=640, height=480, fg_color="#2a2f37")
        self.video.pack(side="left", padx=12, pady=12)
        self.panel = ctk.CTkFrame(body, fg_color="#1d2633", corner_radius=8, width=350)
        self.panel.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        self.panel.pack_propagate(False)
        self.lbl_title = ctk.CTkLabel(self.panel, text="", font=("Arial", 22, "bold"), wraplength=320)
        self.lbl_title.pack(pady=(70, 8))
        self.lbl_name = ctk.CTkLabel(self.panel, text="", font=("Arial", 28, "bold"), wraplength=320)
        self.lbl_name.pack(pady=4)
        self.lbl_type = ctk.CTkLabel(self.panel, text="", font=("Arial", 30, "bold"), text_color="#5fe0a6")
        self.lbl_type.pack(pady=4)
        self.prog = ctk.CTkProgressBar(self.panel, width=240)
        self.prog.set(0)
        self.lbl_hint = ctk.CTkLabel(self.panel, text="", font=("Arial", 15), text_color="#b6c0cc", wraplength=320)
        self.lbl_hint.pack(pady=8)
        self.lbl_recent = ctk.CTkLabel(self.panel, text="", font=("Arial", 13), text_color="#9aa4b2", justify="left")
        self.lbl_recent.pack(side="bottom", pady=12)
        self._photo = None
        self._t_recent = 0

    def close(self):
        self.app.stop.set()
        self.root.quit()

    def tick(self):
        m = self.app.poll()
        st = m.get("state", "idle")
        bg, color, title = STYLE.get(st, STYLE["idle"])
        f = self.app.latest
        if f is not None:
            f = f.copy()
            if m.get("bbox") is not None:
                x, y, w, h = [int(v) for v in m["bbox"]]
                cv2.rectangle(f, (x, y), (x + w, y + h), color, 3)
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            self._photo = ImageTk.PhotoImage(img)
            self.video._label.configure(image=self._photo)  # nhanh hơn CTkImage cho video
        self.panel.configure(fg_color=bg)
        self.lbl_title.configure(text=title)
        name, typ, hint = "", "", m.get("hint", "")
        if st == "attend":
            name, typ = m.get("name", ""), f'{TYPE_VN.get(m.get("att_type"), "")} · {m.get("time", "")[:5]}'
        elif st == "already":
            name, hint = m.get("name", ""), f'Đã chấm lúc {m.get("already_at", "")}'
        elif st == "unknown":
            hint = "Liên hệ quản trị viên để đăng ký"
        elif st == "spoof":
            hint = "Phát hiện ảnh hoặc màn hình. Không chấm công, sự kiện đã được ghi lại."
        elif st == "next":
            hint = ""
        elif st == "idle" and not hint:
            hint = "Khoảng cách 30–50cm, nhìn thẳng"
        self.lbl_name.configure(text=name)
        self.lbl_type.configure(text=typ)
        self.lbl_hint.configure(text=hint)
        if st == "tracking":
            if not self.prog.winfo_ismapped():
                self.prog.pack(pady=6, after=self.lbl_title)
            self.prog.set(m.get("progress", 0))
        elif self.prog.winfo_ismapped():
            self.prog.pack_forget()
        s = m.get("stats", {})
        self.lbl_status.configure(
            text=f'● {self.app.cfg["device"]["name"]} ({self.app.cfg["device"]["id"]})    '
                 f'Camera {self.app.cam_fps:.0f}FPS    Suy luận {s.get("inference_per_s", "-")}/s    '
                 f'Đã đăng ký {s.get("gallery", "-")} mẫu    Hàng đợi {s.get("pending", "-")}')
        self.lbl_clock.configure(text=time.strftime("%H:%M:%S  %d/%m/%Y"))
        if time.time() - self._t_recent > 2:
            self._t_recent = time.time()
            rows = self.db.recent(5)
            self.lbl_recent.configure(text="Gần đây:\n" + "\n".join(
                f'{datetime.fromisoformat(r[0].replace("Z", "+00:00")).astimezone():%H:%M}  '
                f'{TYPE_VN.get(r[2], r[2]):3}  {r[1]}' for r in rows) if rows else "")
        if not self.app.stop.is_set():
            self.root.after(self.period, self.tick)
        else:
            self.root.quit()

    def run(self):
        self.root.after(200, self.tick)
        self.root.mainloop()
