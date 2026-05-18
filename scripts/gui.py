import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import os
import threading
import queue
import time
import pickle

import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageTk

from database import (
    init_db,
    get_attendance_records,
    get_recent_attendance,
    insert_attendance,
    clear_attendance_records
)

from people_manager import (
    get_people_list,
    add_or_update_person,
    get_next_image_path,
    get_person_image_count,
    load_people,
    delete_person
)

from model_trainer import train_model, get_model_info

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CV_SCALER = 4
PROCESS_EVERY_N_FRAMES = 8
TOLERANCE = 0.50

UI_TARGET_FPS = 12
UI_FRAME_INTERVAL = 1 / UI_TARGET_FPS

CONFIRM_SECONDS = 3
COOLDOWN_SECONDS = 10


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống điểm danh bằng nhận dạng khuôn mặt")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 600)

        init_db()

        self.people_map = load_people()

        # Camera thread variables
        self.camera_running = False
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_camera_image = None
        self.camera_capture = None
        self.app_closing = False

        # Capture window variables
        self.capture_window = None
        self.capture_label = None
        self.capture_info_label = None
        self.capture_running = False
        self.capture_thread = None
        self.capture_queue = queue.Queue(maxsize=2)
        self.capture_current_frame = None
        self.capture_current_image = None
        self.capture_raw_name = None
        self.capture_display_name = None

        self.capture_saved_count = 0
        self.dataset_changed = False

        # Train model variables
        self.train_running = False
        self.train_thread = None
        self.train_queue = queue.Queue()

        self.setup_style()
        self.create_widgets()

        self.load_history()
        self.refresh_recent_attendance()
        self.auto_refresh_recent_attendance()
        self.load_people_list()
        self.update_model_status()

        # Bắt đầu vòng lặp đọc queue cho UI
        self.process_camera_queue()
        self.process_train_queue()

        # Xử lý khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_style(self):
        self.colors = {
            "bg": "#f4f6f8",
            "panel": "#ffffff",
            "primary": "#2563eb",
            "primary_dark": "#1d4ed8",
            "success": "#16a34a",
            "warning": "#f59e0b",
            "danger": "#dc2626",
            "muted": "#e5e7eb",
            "text": "#111827",
            "subtext": "#4b5563",
            "table_alt": "#f9fafb",
        }

        self.root.configure(bg=self.colors["bg"])

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background=self.colors["bg"]
        )

        style.configure(
            "TLabelframe",
            background=self.colors["bg"],
            borderwidth=1,
            relief="solid"
        )

        style.configure(
            "TLabelframe.Label",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Arial", 11, "bold")
        )

        style.configure(
            "TNotebook",
            background=self.colors["bg"],
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            padding=(18, 10),
            font=("Arial", 11, "bold")
        )

        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["primary"])],
            foreground=[("selected", "white")]
        )

        style.configure(
            "Title.TLabel",
            font=("Arial", 18, "bold"),
            background=self.colors["bg"],
            foreground=self.colors["text"]
        )

        style.configure(
            "Header.TLabel",
            font=("Arial", 12, "bold"),
            background=self.colors["bg"],
            foreground=self.colors["text"]
        )

        style.configure(
            "Status.TLabel",
            font=("Arial", 11),
            background=self.colors["bg"],
            foreground=self.colors["subtext"]
        )

        style.configure(
            "TButton",
            font=("Arial", 10, "bold"),
            padding=7
        )

        style.configure(
            "Primary.TButton",
            font=("Arial", 10, "bold"),
            padding=7,
            background=self.colors["primary"],
            foreground="white"
        )

        style.map(
            "Primary.TButton",
            background=[("active", self.colors["primary_dark"])]
        )

        style.configure(
            "Danger.TButton",
            font=("Arial", 10, "bold"),
            padding=7,
            background=self.colors["danger"],
            foreground="white"
        )

        style.configure(
            "Success.TButton",
            font=("Arial", 10, "bold"),
            padding=7,
            background=self.colors["success"],
            foreground="white"
        )

        style.configure(
            "Treeview",
            rowheight=28,
            font=("Arial", 10),
            background="white",
            fieldbackground="white",
            foreground=self.colors["text"]
        )

        style.configure(
            "Treeview.Heading",
            font=("Arial", 10, "bold"),
            background=self.colors["muted"],
            foreground=self.colors["text"]
        )

    def update_button_states(self):
        if self.camera_running:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

        if hasattr(self, "train_button"):
            if self.train_running:
                self.train_button.configure(state="disabled")
            else:
                self.train_button.configure(state="normal")

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_attendance = ttk.Frame(self.notebook)
        self.tab_faces = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_attendance, text="Điểm danh")
        self.notebook.add(self.tab_faces, text="Quản lý khuôn mặt")
        self.notebook.add(self.tab_history, text="Lịch sử chấm công")

        self.create_attendance_tab()
        self.create_faces_tab()
        self.create_history_tab()

    # ==================================================
    # TAB 1: ĐIỂM DANH
    # ==================================================

    def create_attendance_tab(self):
        main_frame = ttk.Frame(self.tab_attendance, padding=12)
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        title = ttk.Label(
            main_frame,
            text="Màn hình điểm danh",
            style="Title.TLabel"
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # Khung camera
        camera_frame = ttk.LabelFrame(main_frame, text="Camera", padding=10)
        camera_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        camera_frame.rowconfigure(0, weight=1)
        camera_frame.columnconfigure(0, weight=1)

        self.camera_container = tk.Frame(
            camera_frame,
            bg="black",
            width=720,
            height=480
        )
        self.camera_container.grid(row=0, column=0, sticky="nsew")

        # Không cho frame tự đổi kích thước theo ảnh bên trong
        self.camera_container.grid_propagate(False)

        self.camera_label = tk.Label(
            self.camera_container,
            text="Camera feed sẽ hiển thị ở đây",
            bg="black",
            fg="white",
            font=("Arial", 16)
        )
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        # Khung bên phải
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew")
        right_frame.rowconfigure(2, weight=1)
        right_frame.columnconfigure(0, weight=1)

        status_frame = ttk.LabelFrame(right_frame, text="Trạng thái hệ thống", padding=10)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.fps_label = ttk.Label(
            status_frame,
            text="FPS: --",
            style="Status.TLabel"
        )
        self.fps_label.pack(anchor="w", pady=4)

        self.status_label = tk.Label(
            status_frame,
            text="Status: Chưa khởi động camera",
            anchor="w",
            padx=10,
            pady=8,
            bg=self.colors["muted"],
            fg=self.colors["text"],
            font=("Arial", 11, "bold")
        )
        self.status_label.pack(fill="x", pady=4)

        button_frame = ttk.LabelFrame(right_frame, text="Điều khiển", padding=10)
        button_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.start_button = ttk.Button(
            button_frame,
            text="Start Camera",
            style="Success.TButton",
            command=self.on_start_camera
        )
        self.start_button.pack(side="left", padx=(0, 8))

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Camera",
            style="Danger.TButton",
            command=self.on_stop_camera
        )
        self.stop_button.pack(side="left")
        self.stop_button.configure(state="disabled")

        recent_frame = ttk.LabelFrame(right_frame, text="Điểm danh gần nhất", padding=10)
        recent_frame.grid(row=2, column=0, sticky="nsew")

        columns = ("id", "name", "time", "type")
        self.recent_tree = ttk.Treeview(
            recent_frame,
            columns=columns,
            show="headings",
            height=10
        )

        self.recent_tree.heading("id", text="ID")
        self.recent_tree.heading("name", text="Tên")
        self.recent_tree.heading("time", text="Thời gian")
        self.recent_tree.heading("type", text="Loại")

        self.recent_tree.column("id", width=50, anchor="center")
        self.recent_tree.column("name", width=140)
        self.recent_tree.column("time", width=160)
        self.recent_tree.column("type", width=70, anchor="center")

        self.recent_tree.pack(fill="both", expand=True)

    # ==================================================
    # TAB 2: QUẢN LÝ KHUÔN MẶT
    # ==================================================

    def create_faces_tab(self):
        main_frame = ttk.Frame(self.tab_faces, padding=12)
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        title = ttk.Label(
            main_frame,
            text="Quản lý khuôn mặt",
            style="Title.TLabel"
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        form_frame = ttk.LabelFrame(main_frame, text="Thêm người mới", padding=12)
        form_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(form_frame, text="Tên thư mục / mã người dùng:").pack(anchor="w")
        self.raw_name_entry = ttk.Entry(form_frame)
        self.raw_name_entry.pack(fill="x", pady=(4, 12))
        self.raw_name_entry.insert(0, "nguyen_van_a")

        ttk.Label(form_frame, text="Tên hiển thị:").pack(anchor="w")
        self.display_name_entry = ttk.Entry(form_frame)
        self.display_name_entry.pack(fill="x", pady=(4, 12))
        self.display_name_entry.insert(0, "Nguyen Van A")

        ttk.Button(
            form_frame,
            text="Lưu người / Cập nhật tên",
            command=self.on_save_person
        ).pack(fill="x", pady=4)

        ttk.Button(
            form_frame,
            text="Chụp ảnh khuôn mặt",
            command=self.on_capture_images
        ).pack(fill="x", pady=4)

        self.train_button = ttk.Button(
            form_frame,
            text="Train lại model",
            style="Primary.TButton",
            command=self.on_train_model
        )
        self.train_button.pack(fill="x", pady=4)
        
        self.train_status_label = ttk.Label(
            form_frame,
            text="Train status: Chưa train",
            style="Status.TLabel",
            wraplength=300
        )
        self.train_status_label.pack(fill="x", pady=(12, 4))

        list_frame = ttk.LabelFrame(main_frame, text="Danh sách người đã đăng ký", padding=12)
        list_frame.grid(row=1, column=1, sticky="nsew")

        columns = ("raw_name", "display_name", "image_count")
        self.people_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings"
        )

        self.people_tree.heading("raw_name", text="Tên thư mục")
        self.people_tree.heading("display_name", text="Tên hiển thị")
        self.people_tree.heading("image_count", text="Số ảnh")

        self.people_tree.column("raw_name", width=180)
        self.people_tree.column("display_name", width=220)
        self.people_tree.column("image_count", width=80, anchor="center")

        self.people_tree.pack(fill="both", expand=True)

        people_button_frame = ttk.Frame(list_frame)
        people_button_frame.pack(fill="x", pady=(8, 0))

        ttk.Button(
            people_button_frame,
            text="Chọn để sửa",
            command=self.on_load_selected_person
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            people_button_frame,
            text="Xóa người được chọn",
            command=self.on_delete_selected_person
        ).pack(side="left")

        self.people_tree.bind(
            "<Double-1>",
            lambda event: self.on_load_selected_person()
        )

    # ==================================================
    # TAB 3: LỊCH SỬ CHẤM CÔNG
    # ==================================================

    def create_history_tab(self):
        main_frame = ttk.Frame(self.tab_history, padding=12)
        main_frame.pack(fill="both", expand=True)

        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(0, weight=1)

        title = ttk.Label(
            main_frame,
            text="Lịch sử chấm công",
            style="Title.TLabel"
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        filter_frame = ttk.LabelFrame(main_frame, text="Bộ lọc", padding=10)
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(filter_frame, text="Tên (có thể nhập không dấu):").pack(side="left", padx=(0, 4))
        self.filter_name_entry = ttk.Entry(filter_frame, width=25)
        self.filter_name_entry.pack(side="left", padx=(0, 12))

        ttk.Label(filter_frame, text="Ngày YYYY-MM-DD:").pack(side="left", padx=(0, 4))
        self.filter_date_entry = ttk.Entry(filter_frame, width=18)
        self.filter_date_entry.pack(side="left", padx=(0, 12))

        ttk.Button(
            filter_frame,
            text="Lọc",
            command=self.on_filter_history
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            filter_frame,
            text="Refresh",
            command=self.on_refresh_history
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            filter_frame,
            text="Export CSV",
            command=self.on_export_csv
        ).pack(side="left")

        ttk.Button(
            filter_frame,
            text="Xóa lịch sử",
            command=self.on_clear_history
        ).pack(side="left", padx=(8, 0))

        table_frame = ttk.LabelFrame(main_frame, text="Danh sách bản ghi", padding=10)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("id", "name", "timestamp", "type")
        self.history_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        self.history_tree.heading("id", text="ID")
        self.history_tree.heading("name", text="Tên")
        self.history_tree.heading("timestamp", text="Thời gian")
        self.history_tree.heading("type", text="Loại")

        self.history_tree.column("id", width=60, anchor="center")
        self.history_tree.column("name", width=220)
        self.history_tree.column("timestamp", width=200)
        self.history_tree.column("type", width=80, anchor="center")

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        self.history_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def load_people_list(self):
        try:
            self.people_map = load_people()
            people = get_people_list()

            self.clear_treeview(self.people_tree)

            for person in people:
                self.people_tree.insert(
                    "",
                    "end",
                    values=(
                        person["raw_name"],
                        person["display_name"],
                        person["image_count"]
                    )
                )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải danh sách người:\n{e}")

    def get_selected_person(self):
        selection = self.people_tree.selection()

        if not selection:
            messagebox.showwarning(
                "Cảnh báo",
                "Vui lòng chọn một người trong danh sách."
            )
            return None

        values = self.people_tree.item(selection[0], "values")

        if not values:
            messagebox.showwarning(
                "Cảnh báo",
                "Không đọc được dữ liệu người được chọn."
            )
            return None

        raw_name = values[0]
        display_name = values[1]

        return raw_name, display_name

    # ==================================================
    # DATABASE / TABLE FUNCTIONS
    # ==================================================

    def clear_treeview(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def load_history(self):
        name_filter = self.filter_name_entry.get().strip()
        date_filter = self.filter_date_entry.get().strip()

        try:
            rows = get_attendance_records(
                name_filter=name_filter,
                date_filter=date_filter
            )

            self.clear_treeview(self.history_tree)

            for row in rows:
                self.history_tree.insert("", "end", values=row)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải lịch sử chấm công:\n{e}")

    def refresh_recent_attendance(self):
        try:
            rows = get_recent_attendance(limit=10)

            self.clear_treeview(self.recent_tree)

            for row in rows:
                self.recent_tree.insert("", "end", values=row)

        except Exception as e:
            print(f"[ERROR] Không thể tải điểm danh gần nhất: {e}")

    def auto_refresh_recent_attendance(self):
        self.refresh_recent_attendance()
        self.root.after(3000, self.auto_refresh_recent_attendance)

    def export_history_to_csv(self):
        name_filter = self.filter_name_entry.get().strip()
        date_filter = self.filter_date_entry.get().strip()

        try:
            rows = get_attendance_records(
                name_filter=name_filter,
                date_filter=date_filter
            )

            if not rows:
                messagebox.showinfo("Thông báo", "Không có dữ liệu để xuất.")
                return

            file_path = filedialog.asksaveasfilename(
                title="Lưu file CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_path:
                return

            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Tên", "Thời gian", "Loại"])
                writer.writerows(rows)

            messagebox.showinfo("Thành công", f"Đã xuất file:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể export CSV:\n{e}")

    def clear_frame_queue(self):
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

    def get_display_name(self, raw_name):
        return self.people_map.get(raw_name, raw_name)

    def handle_attendance(self, detected_raw_names, confirmed_time, cooldown_time):
        current_time = time.time()

        detected_raw_names = [
            name for name in detected_raw_names
            if name != "Unknown"
        ]

        detected_set = set(detected_raw_names)

        # Nếu mất mặt giữa chừng thì reset thời gian xác nhận
        for name in list(confirmed_time.keys()):
            if name not in detected_set:
                confirmed_time.pop(name, None)

        if not detected_set:
            return None, False

        for raw_name in detected_set:
            display_name = self.get_display_name(raw_name)

            last_cooldown = cooldown_time.get(raw_name, 0)
            if current_time - last_cooldown < COOLDOWN_SECONDS:
                remain = int(COOLDOWN_SECONDS - (current_time - last_cooldown))
                return f"{display_name} đang cooldown, còn {remain}s", False

            if raw_name not in confirmed_time:
                confirmed_time[raw_name] = current_time
                return f"Đang xác nhận {display_name}...", False

            elapsed_confirm = current_time - confirmed_time[raw_name]

            if elapsed_confirm >= CONFIRM_SECONDS:
                attendance_type, timestamp = insert_attendance(display_name)

                cooldown_time[raw_name] = current_time
                confirmed_time.pop(raw_name, None)

                return f"{display_name} - {attendance_type} - {timestamp}", True

            remain = CONFIRM_SECONDS - elapsed_confirm
            return f"Giữ khuôn mặt {display_name} thêm {remain:.1f}s", False

        return None, False

    def draw_face_results(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= CV_SCALER
            right *= CV_SCALER
            bottom *= CV_SCALER
            left *= CV_SCALER

            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
            cv2.rectangle(
                frame,
                (left - 3, top - 35),
                (right + 3, top),
                (244, 42, 3),
                cv2.FILLED
            )

            cv2.putText(
                frame,
                name,
                (left + 6, top - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                1
            )

        return frame

    # ==================================================
    # CAMERA FUNCTIONS
    # ==================================================

    def start_camera(self):
        if self.camera_running:
            return

        self.camera_running = True
        self.set_system_status("Đang khởi động camera...", "warning")
        self.update_button_states()

        self.camera_thread = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )
        self.camera_thread.start()

    def stop_camera(self):
        if not self.camera_running:
            self.clear_frame_queue()
            self.fps_label.config(text="FPS: --")
            self.set_system_status("Camera đã dừng", "info")
            self.update_button_states()
            return

        self.camera_running = False
        self.set_system_status("Đang dừng camera...", "warning")

        # Đợi camera thread thoát nhẹ nhàng
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.5)

        self.clear_frame_queue()

        self.camera_thread = None
        self.camera_capture = None

        self.fps_label.config(text="FPS: --")
        self.set_system_status("Đã dừng camera", "info")
        self.update_button_states()

        self.camera_label.config(
            image="",
            text="Camera đã dừng",
            bg="black",
            fg="white"
        )
        self.current_camera_image = None

    def camera_worker(self):
        cap = None

        try:
            if not os.path.exists(ENCODINGS_PATH):
                self.frame_queue.put({
                    "type": "error",
                    "message": "Không tìm thấy encodings.pickle. Hãy train model trước."
                })
                self.camera_running = False
                return

            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.loads(f.read())

            known_face_encodings = data["encodings"]
            known_face_names = data["names"]

            if len(known_face_encodings) == 0:
                self.frame_queue.put({
                    "type": "error",
                    "message": "encodings.pickle không có dữ liệu khuôn mặt."
                })
                self.camera_running = False
                return

            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.camera_capture = cap

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                self.frame_queue.put({
                    "type": "error",
                    "message": "Không mở được webcam"
                })
                self.camera_running = False
                return

            warmup_ok = False
            for _ in range(10):
                if not self.camera_running:
                    return

                ret, frame = cap.read()
                if ret:
                    warmup_ok = True
                    break

                time.sleep(0.2)

            if not warmup_ok:
                self.frame_queue.put({
                    "type": "error",
                    "message": "Webcam mở được nhưng không đọc được frame"
                })
                self.camera_running = False
                return

            frame_count = 0
            start_time = time.time()
            fps = 0
            process_counter = 0
            last_ui_update_time = 0

            face_locations = []
            face_names = []

            confirmed_time = {}
            cooldown_time = {}

            current_status = "Camera đang chạy"

            self.frame_queue.put({
                "type": "status",
                "message": "Camera đang chạy"
            })

            while self.camera_running:
                ret, frame = cap.read()

                if not ret:
                    self.frame_queue.put({
                        "type": "error",
                        "message": "Không đọc được frame từ webcam"
                    })
                    self.camera_running = False
                    break

                attendance_saved = False

                if process_counter % PROCESS_EVERY_N_FRAMES == 0:
                    resized_frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=(1 / CV_SCALER),
                        fy=(1 / CV_SCALER)
                    )

                    rgb_resized_frame = cv2.cvtColor(
                        resized_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    face_locations = face_recognition.face_locations(
                        rgb_resized_frame,
                        number_of_times_to_upsample=0,
                        model="hog"
                    )

                    face_encodings = face_recognition.face_encodings(
                        rgb_resized_frame,
                        face_locations,
                        num_jitters=1,
                        model="small"
                    )

                    face_names = []
                    raw_names_for_attendance = []

                    for face_encoding in face_encodings:
                        raw_name = "Unknown"

                        face_distances = face_recognition.face_distance(
                            known_face_encodings,
                            face_encoding
                        )

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            best_distance = face_distances[best_match_index]

                            if best_distance < TOLERANCE:
                                raw_name = known_face_names[best_match_index]

                        display_name = self.get_display_name(raw_name)

                        face_names.append(display_name)
                        raw_names_for_attendance.append(raw_name)

                    status, attendance_saved = self.handle_attendance(
                        raw_names_for_attendance,
                        confirmed_time,
                        cooldown_time
                    )

                    if status:
                        current_status = status
                    elif len(raw_names_for_attendance) == 0:
                        current_status = "Camera đang chạy"

                process_counter += 1

                display_frame = frame.copy()
                display_frame = self.draw_face_results(
                    display_frame,
                    face_locations,
                    face_names
                )

                frame_count += 1
                elapsed = time.time() - start_time

                if elapsed >= 1:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

                current_time = time.time()

                if current_time - last_ui_update_time >= UI_FRAME_INTERVAL:
                    last_ui_update_time = current_time

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.frame_queue.put({
                        "type": "frame",
                        "frame": display_frame,
                        "fps": fps,
                        "status": current_status,
                        "attendance_saved": attendance_saved
                    })

        except Exception as e:
            self.frame_queue.put({
                "type": "error",
                "message": f"Lỗi camera thread: {e}"
            })
            self.camera_running = False

        finally:
            if cap is not None:
                cap.release()

            self.camera_capture = None

    def process_camera_queue(self):
        try:
            while True:
                item = self.frame_queue.get_nowait()

                if item["type"] == "frame":
                    if not self.camera_running:
                        continue

                    frame = item["frame"]
                    fps = item["fps"]
                    status = item.get("status", "Camera đang chạy")

                    self.update_camera_frame(frame)
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    status_lower = status.lower()

                    if item.get("attendance_saved", False):
                        level = "success"
                    elif "cooldown" in status_lower or "xác nhận" in status_lower or "giữ khuôn mặt" in status_lower:
                        level = "warning"
                    else:
                        level = "info"

                    self.set_system_status(status, level)

                    if item.get("attendance_saved", False):
                        self.refresh_recent_attendance()
                        self.load_history()

                elif item["type"] == "status":
                    if not self.camera_running:
                        continue

                    self.set_system_status(item["message"], "info")

                elif item["type"] == "error":
                    self.camera_running = False
                    self.update_button_states()
                    self.set_system_status(f"Lỗi - {item['message']}", "error")

                    if not self.app_closing:
                        messagebox.showerror("Lỗi camera", item["message"])

        except queue.Empty:
            pass

        self.root.after(30, self.process_camera_queue)

    def update_camera_frame(self, frame):
        # Resize frame theo kích thước camera_label
        label_width = self.camera_container.winfo_width()
        label_height = self.camera_container.winfo_height()

        if label_width <= 1 or label_height <= 1:
            label_width = 640
            label_height = 480

        frame_height, frame_width = frame.shape[:2]

        scale = min(
            label_width / frame_width,
            label_height / frame_height
        )

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # BGR OpenCV → RGB Pillow
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)

        self.camera_label.config(image=photo, text="")
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")
        self.current_camera_image = photo

    def on_close(self):
        if self.train_running:
            confirm = messagebox.askyesno(
                "Đang train model",
                "Model đang được train. Đóng ứng dụng lúc này có thể làm gián đoạn quá trình train.\n\nBạn vẫn muốn đóng?"
            )

            if not confirm:
                return

        self.app_closing = True

        if self.capture_running:
            self.stop_capture_camera()

        if self.camera_running:
            self.stop_camera()

        self.root.destroy()

    def set_system_status(self, message, level="info"):
        color_map = {
            "info": self.colors["muted"],
            "success": self.colors["success"],
            "warning": self.colors["warning"],
            "error": self.colors["danger"],
        }

        fg_map = {
            "info": self.colors["text"],
            "success": "white",
            "warning": "black",
            "error": "white",
        }

        bg = color_map.get(level, self.colors["muted"])
        fg = fg_map.get(level, self.colors["text"])

        self.status_label.config(
            text=f"Status: {message}",
            bg=bg,
            fg=fg
        )

    # ==================================================
    # CAPTURE FACE IMAGE FUNCTIONS
    # ==================================================

    def open_capture_window(self, raw_name, display_name):
        if self.capture_window is not None and self.capture_window.winfo_exists():
            self.capture_window.lift()
            return

        self.capture_raw_name = raw_name
        self.capture_display_name = display_name
        self.capture_current_frame = None
        self.capture_saved_count = 0

        self.capture_window = tk.Toplevel(self.root)
        self.capture_window.title(f"Chụp ảnh khuôn mặt - {display_name}")
        self.capture_window.geometry("760x600")
        self.capture_window.minsize(700, 520)

        self.capture_window.protocol("WM_DELETE_WINDOW", self.close_capture_window)
        self.capture_window.bind("<space>", lambda event: self.save_capture_image())
        self.capture_window.bind("<Escape>", lambda event: self.close_capture_window())

        main_frame = ttk.Frame(self.capture_window, padding=12)
        main_frame.pack(fill="both", expand=True)

        title = ttk.Label(
            main_frame,
            text=f"Chụp ảnh cho: {display_name}",
            style="Title.TLabel"
        )
        title.pack(anchor="w", pady=(0, 10))

        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding=10)
        preview_frame.pack(fill="both", expand=True)

        self.capture_preview_container = tk.Frame(
            preview_frame,
            bg="black",
            width=640,
            height=420
        )
        self.capture_preview_container.pack(fill="both", expand=True)
        self.capture_preview_container.pack_propagate(False)

        self.capture_label = tk.Label(
            self.capture_preview_container,
            text="Đang mở camera...",
            bg="black",
            fg="white",
            font=("Arial", 16)
        )
        self.capture_label.place(relx=0.5, rely=0.5, anchor="center")

        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill="x", pady=(10, 0))

        image_count = get_person_image_count(raw_name)

        self.capture_info_label = ttk.Label(
            info_frame,
            text=f"Raw name: {raw_name} | Số ảnh hiện tại: {image_count}",
            style="Status.TLabel"
        )
        self.capture_info_label.pack(side="left")

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(
            button_frame,
            text="Chụp ảnh",
            command=self.save_capture_image
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            button_frame,
            text="Đóng",
            command=self.close_capture_window
        ).pack(side="left")

        self.start_capture_camera()
        self.process_capture_queue()

    def start_capture_camera(self):
        if self.capture_running:
            return

        self.capture_running = True

        self.capture_thread = threading.Thread(
            target=self.capture_camera_worker,
            daemon=True
        )
        self.capture_thread.start()

    def stop_capture_camera(self):
        if not self.capture_running:
            return

        self.capture_running = False

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)

        self.capture_thread = None

    def capture_camera_worker(self):
        cap = None

        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                self.capture_queue.put({
                    "type": "error",
                    "message": "Không mở được webcam để chụp ảnh"
                })
                self.capture_running = False
                return

            warmup_ok = False

            for _ in range(10):
                if not self.capture_running:
                    return

                ret, frame = cap.read()

                if ret:
                    warmup_ok = True
                    break

                time.sleep(0.2)

            if not warmup_ok:
                self.capture_queue.put({
                    "type": "error",
                    "message": "Webcam mở được nhưng không đọc được frame"
                })
                self.capture_running = False
                return

            while self.capture_running:
                ret, frame = cap.read()

                if not ret:
                    self.capture_queue.put({
                        "type": "error",
                        "message": "Không đọc được frame từ webcam"
                    })
                    self.capture_running = False
                    break

                if self.capture_queue.full():
                    try:
                        self.capture_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.capture_queue.put({
                    "type": "frame",
                    "frame": frame
                })

                time.sleep(0.01)

        except Exception as e:
            self.capture_queue.put({
                "type": "error",
                "message": f"Lỗi capture camera: {e}"
            })
            self.capture_running = False

        finally:
            if cap is not None:
                cap.release()

    def process_capture_queue(self):
        if self.capture_window is None or not self.capture_window.winfo_exists():
            return

        try:
            while True:
                item = self.capture_queue.get_nowait()

                if item["type"] == "frame":
                    if not self.capture_running:
                        continue

                    frame = item["frame"]
                    self.capture_current_frame = frame.copy()
                    self.update_capture_preview(frame)

                elif item["type"] == "error":
                    self.capture_running = False
                    messagebox.showerror("Lỗi camera", item["message"])

        except queue.Empty:
            pass

        self.capture_window.after(30, self.process_capture_queue)

    def update_capture_preview(self, frame):
        if self.capture_label is None:
            return

        label_width = self.capture_preview_container.winfo_width()
        label_height = self.capture_preview_container.winfo_height()

        if label_width <= 1 or label_height <= 1:
            label_width = 640
            label_height = 420

        frame_height, frame_width = frame.shape[:2]

        scale = min(
            label_width / frame_width,
            label_height / frame_height
        )

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)

        self.capture_label.config(image=photo, text="")
        self.capture_label.place(relx=0.5, rely=0.5, anchor="center")

        self.capture_current_image = photo

    def save_capture_image(self):
        if self.capture_current_frame is None:
            messagebox.showwarning(
                "Cảnh báo",
                "Chưa có frame để lưu.",
                parent=self.capture_window
            )
            return

        try:
            file_path = get_next_image_path(self.capture_raw_name)

            success = cv2.imwrite(file_path, self.capture_current_frame)

            if not success:
                messagebox.showerror(
                    "Lỗi",
                    "Không thể lưu ảnh.",
                    parent=self.capture_window
                )
                return

            image_count = get_person_image_count(self.capture_raw_name)
            filename = os.path.basename(file_path)

            self.capture_saved_count += 1
            self.dataset_changed = True

            if self.capture_info_label is not None:
                self.capture_info_label.config(
                    text=(
                        f"Raw name: {self.capture_raw_name} | "
                        f"Số ảnh hiện tại: {image_count} | "
                        f"Đã lưu: {filename} | "
                        f"Cần train lại model"
                    )
                )

            self.load_people_list()

            # Giữ cửa sổ capture ở phía trước
            if self.capture_window is not None and self.capture_window.winfo_exists():
                self.capture_window.lift()
                self.capture_window.focus_force()

        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể lưu ảnh:\n{e}",
                parent=self.capture_window
            )

    def close_capture_window(self):
        should_train = False

        if self.capture_saved_count > 0:
            should_train = messagebox.askyesno(
                "Train lại model?",
                f"Bạn vừa chụp {self.capture_saved_count} ảnh mới.\n\n"
                f"Cần train lại model để ảnh mới có hiệu lực trong nhận diện.\n\n"
                f"Bạn có muốn train lại model ngay không?",
                parent=self.capture_window
            )

        self.stop_capture_camera()

        if self.capture_window is not None and self.capture_window.winfo_exists():
            self.capture_window.destroy()

        self.capture_window = None
        self.capture_label = None
        self.capture_info_label = None
        self.capture_current_frame = None
        self.capture_current_image = None
        self.capture_saved_count = 0

        if should_train:
            self.start_train_model()

    def update_model_status(self):
        info = get_model_info()

        if not info["exists"]:
            self.train_status_label.config(
                text="Train status: Chưa có model. Vui lòng train model."
            )
            return

        if info["error"]:
            self.train_status_label.config(
                text=f"Train status: Model lỗi - {info['error']}"
            )
            return

        self.train_status_label.config(
            text=(
                f"Train status: Đã có model | "
                f"Người: {info['people_count']} | "
                f"Encodings: {info['total_encodings']} | "
                f"Cập nhật: {info['updated_at']}"
            )
        )

    # ==================================================
    # TRAIN MODEL FUNCTIONS
    # ==================================================

    def start_train_model(self):
        if self.train_running:
            messagebox.showinfo("Thông báo", "Model đang được train, vui lòng chờ.")
            return

        # Không nên train khi camera điểm danh đang chạy
        if self.camera_running:
            self.stop_camera()

        # Không nên train khi cửa sổ capture đang chạy
        if self.capture_running:
            self.stop_capture_camera()

        self.train_running = True
        self.update_button_states()
        self.train_status_label.config(text="Train status: Đang train model...")
        self.set_system_status("Đang train model...", "warning")

        self.train_thread = threading.Thread(
            target=self.train_model_worker,
            daemon=True
        )
        self.train_thread.start()

    def train_model_worker(self):
        try:
            def progress_callback(message):
                self.train_queue.put({
                    "type": "progress",
                    "message": message
                })

            result = train_model(progress_callback=progress_callback)

            self.train_queue.put({
                "type": "success",
                "result": result
            })

        except Exception as e:
            self.train_queue.put({
                "type": "error",
                "message": str(e)
            })

    def process_train_queue(self):
        try:
            while True:
                item = self.train_queue.get_nowait()

                if item["type"] == "progress":
                    message = item["message"]
                    self.train_status_label.config(text=f"Train status: {message}")

                elif item["type"] == "success":
                    self.train_running = False
                    self.update_button_states()

                    result = item["result"]

                    message = (
                        "Train model hoàn tất.\n\n"
                        f"Tổng số ảnh: {result['total_images']}\n"
                        f"Số encoding: {result['total_encodings']}\n"
                        f"Ảnh bị bỏ qua: {result['skipped_images']}\n"
                        f"File lưu: {result['encodings_path']}"
                    )

                    self.train_status_label.config(
                        text=(
                            "Train status: Hoàn tất | "
                            f"Encodings: {result['total_encodings']} | "
                            f"Skipped: {result['skipped_images']}"
                        )
                    )

                    self.set_system_status("Train model hoàn tất", "success")

                    # Reload people map để nhận tên hiển thị mới nhất
                    self.people_map = load_people()
                    self.load_people_list()
                    self.update_model_status()

                    messagebox.showinfo("Thành công", message)

                elif item["type"] == "error":
                    self.train_running = False
                    self.update_button_states()

                    self.train_status_label.config(
                        text=f"Train status: Lỗi - {item['message']}"
                    )
                    self.set_system_status("Train model lỗi", "error")

                    messagebox.showerror(
                        "Lỗi train model",
                        f"Không thể train model:\n{item['message']}"
                    )

        except queue.Empty:
            pass

        self.root.after(200, self.process_train_queue)        

    # ==================================================
    # PLACEHOLDER EVENTS
    # ==================================================
    def on_save_person(self):
        raw_name = self.raw_name_entry.get().strip()
        display_name = self.display_name_entry.get().strip()

        try:
            saved_raw_name, saved_display_name = add_or_update_person(
                raw_name,
                display_name
            )

            self.load_people_list()

            messagebox.showinfo(
                "Thành công",
                f"Đã lưu người dùng:\n\n"
                f"Raw name: {saved_raw_name}\n"
                f"Display name: {saved_display_name}"
            )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu người dùng:\n{e}")

    def on_start_camera(self):
        self.start_camera()

    def on_stop_camera(self):
        self.stop_camera()

    def on_capture_images(self):
        raw_name = self.raw_name_entry.get().strip()
        display_name = self.display_name_entry.get().strip()

        try:
            saved_raw_name, saved_display_name = add_or_update_person(
                raw_name,
                display_name
            )

            self.load_people_list()

            # Không cho chạy 2 luồng camera cùng lúc
            if self.camera_running:
                self.stop_camera()

            self.open_capture_window(saved_raw_name, saved_display_name)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở chức năng chụp ảnh:\n{e}")

    def on_train_model(self):
        confirm = messagebox.askyesno(
            "Xác nhận",
            "Bạn có muốn train lại toàn bộ model từ dataset hiện tại không?"
        )

        if not confirm:
            return

        self.start_train_model()

    def on_filter_history(self):
        self.load_history()

    def on_refresh_history(self):
        self.filter_name_entry.delete(0, tk.END)
        self.filter_date_entry.delete(0, tk.END)
        self.load_history()
        self.refresh_recent_attendance()

    def on_export_csv(self):
        self.export_history_to_csv()

    def on_load_selected_person(self):
        selected = self.get_selected_person()

        if selected is None:
            return

        raw_name, display_name = selected

        self.raw_name_entry.delete(0, tk.END)
        self.raw_name_entry.insert(0, raw_name)

        self.display_name_entry.delete(0, tk.END)
        self.display_name_entry.insert(0, display_name)

    def on_delete_selected_person(self):
        selected = self.get_selected_person()

        if selected is None:
            return

        raw_name, display_name = selected

        confirm = messagebox.askyesno(
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa người này khỏi dataset?\n\n"
            f"Raw name: {raw_name}\n"
            f"Display name: {display_name}\n\n"
            f"Thư mục ảnh sẽ được chuyển sang deleted_dataset/ để backup.\n"
            f"Lịch sử chấm công trong attendance.db sẽ không bị xóa."
        )

        if not confirm:
            return

        try:
            # Không nên xóa dataset khi camera/capture đang dùng webcam
            if self.camera_running:
                self.stop_camera()

            if self.capture_running:
                self.stop_capture_camera()

            result = delete_person(raw_name, backup=True)

            self.people_map = load_people()
            self.load_people_list()

            backup_path = result.get("backup_path")

            message = (
                f"Đã xóa người dùng khỏi dataset.\n\n"
                f"Raw name: {result['raw_name']}\n"
                f"Display name: {result['display_name']}\n"
            )

            if backup_path:
                message += f"\nBackup tại:\n{backup_path}"

            messagebox.showinfo("Thành công", message)

            should_train = messagebox.askyesno(
                "Train lại model?",
                "Bạn nên train lại model sau khi xóa người.\n\n"
                "Nếu không train lại, encodings.pickle cũ vẫn có thể còn dữ liệu của người vừa xóa.\n\n"
                "Bạn có muốn train lại model ngay không?"
            )

            if should_train:
                self.start_train_model()

        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể xóa người dùng:\n{e}"
            )

    def on_clear_history(self):
        confirm = messagebox.askyesno(
            "Xác nhận xóa lịch sử",
            "Bạn có chắc muốn xóa toàn bộ lịch sử chấm công không?\n\n"
            "Hành động này sẽ xóa toàn bộ dữ liệu trong bảng attendance "
            "và reset ID về 1."
        )

        if not confirm:
            return

        confirm_again = messagebox.askyesno(
            "Xác nhận lần nữa",
            "Dữ liệu lịch sử sau khi xóa sẽ không thể khôi phục từ database.\n\n"
            "Bạn vẫn muốn tiếp tục?"
        )

        if not confirm_again:
            return

        try:
            clear_attendance_records()

            self.load_history()
            self.refresh_recent_attendance()

            messagebox.showinfo(
                "Thành công",
                "Đã xóa toàn bộ lịch sử chấm công."
            )

        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể xóa lịch sử chấm công:\n{e}"
            )


def main():
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()