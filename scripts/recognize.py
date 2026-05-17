import face_recognition
import cv2
import numpy as np
import time
import pickle
import os

from database import init_db, insert_attendance


# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")

CAMERA_PATH = "/dev/video0"


# =========================
# CẤU HÌNH NHẬN DIỆN
# =========================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cv_scaler = 4
PROCESS_EVERY_N_FRAMES = 5
TOLERANCE = 0.50

CONFIRM_SECONDS = 3
COOLDOWN_SECONDS = 10


# =========================
# MAP TÊN DATASET → TÊN HIỂN THỊ
# =========================

NAME_MAP = {
    "nguyen_van_a": "Nguyen Van A",
}


# =========================
# BIẾN TOÀN CỤC
# =========================

face_locations = []
face_names = []

frame_count = 0
start_time = time.time()
fps = 0
process_counter = 0

confirmed_time = {}
cooldown_time = {}

status_message = "Ready"
status_message_time = 0


# =========================
# HÀM HỖ TRỢ
# =========================

def get_display_name(raw_name):
    return NAME_MAP.get(raw_name, raw_name)


def set_status(message):
    global status_message, status_message_time
    status_message = message
    status_message_time = time.time()
    print(message)


def handle_attendance(detected_raw_names):
    """
    detected_raw_names: danh sách tên raw nhận diện được trong frame hiện tại.
    Ví dụ: ["nguyen_van_a"]
    """

    global confirmed_time, cooldown_time

    current_time = time.time()

    detected_raw_names = [
        name for name in detected_raw_names
        if name != "Unknown"
    ]

    detected_set = set(detected_raw_names)

    # Reset confirmed_time nếu người đó không còn xuất hiện trong frame hiện tại
    for name in list(confirmed_time.keys()):
        if name not in detected_set:
            confirmed_time.pop(name, None)

    for raw_name in detected_set:
        display_name = get_display_name(raw_name)

        # Nếu đang cooldown thì chưa ghi tiếp
        last_cooldown = cooldown_time.get(raw_name, 0)
        if current_time - last_cooldown < COOLDOWN_SECONDS:
            remain = int(COOLDOWN_SECONDS - (current_time - last_cooldown))
            set_status(f"[WAIT] {display_name} đang cooldown, còn {remain}s")
            continue

        # Nếu mới thấy người này, bắt đầu đếm thời gian xác nhận
        if raw_name not in confirmed_time:
            confirmed_time[raw_name] = current_time
            set_status(f"[INFO] Đang xác nhận {display_name}...")
            continue

        elapsed_confirm = current_time - confirmed_time[raw_name]

        # Nếu đã nhận diện liên tục đủ CONFIRM_SECONDS thì ghi chấm công
        if elapsed_confirm >= CONFIRM_SECONDS:
            attendance_type, timestamp = insert_attendance(display_name)

            set_status(
                f"[SUCCESS] {display_name} - {attendance_type} - {timestamp}"
            )

            cooldown_time[raw_name] = current_time
            confirmed_time.pop(raw_name, None)
        else:
            remain = CONFIRM_SECONDS - elapsed_confirm
            set_status(f"[INFO] Giữ khuôn mặt {display_name} thêm {remain:.1f}s")


def calculate_fps():
    global frame_count, start_time, fps

    frame_count += 1
    elapsed_time = time.time() - start_time

    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    return fps


def process_frame(frame):
    global face_locations, face_names

    resized_frame = cv2.resize(
        frame,
        (0, 0),
        fx=(1 / cv_scaler),
        fy=(1 / cv_scaler)
    )

    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

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

        display_name = get_display_name(raw_name)

        face_names.append(display_name)
        raw_names_for_attendance.append(raw_name)

    handle_attendance(raw_names_for_attendance)


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(
            frame,
            (left - 3, top - 35),
            (right + 3, top),
            (244, 42, 3),
            cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame,
            name,
            (left + 6, top - 6),
            font,
            0.7,
            (255, 255, 255),
            1
        )

    return frame


def draw_status(frame):
    current_fps = calculate_fps()

    cv2.putText(
        frame,
        f"FPS: {current_fps:.1f}",
        (frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Hiển thị status trong khoảng 4 giây gần nhất
    if time.time() - status_message_time < 4:
        cv2.putText(
            frame,
            status_message,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return frame


# =========================
# MAIN
# =========================

print("[INFO] initializing database...")
init_db()

print("[INFO] loading encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]

if len(known_face_encodings) == 0:
    print("[ERROR] Không có dữ liệu khuôn mặt trong encodings.pickle")
    exit()

print(f"[INFO] Loaded {len(known_face_encodings)} face encodings")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set MJPG trước, sau đó mới set độ phân giải
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Không dùng dòng này vì webcam của bạn có thể bị timeout
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Không mở được webcam")
    exit()

# Warm-up camera: đọc thử vài frame trước khi vào vòng lặp chính
for i in range(10):
    ret, frame = cap.read()

    if ret:
        print("[INFO] Webcam warm-up OK")
        break

    print(f"[WARN] Warm-up frame {i + 1}/10 failed")
    time.sleep(0.2)
else:
    print("[ERROR] Webcam mở được nhưng không đọc được frame sau warm-up")
    cap.release()
    exit()

print("[INFO] starting video stream...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Không đọc được frame từ webcam!")
        break

    if process_counter % PROCESS_EVERY_N_FRAMES == 0:
        process_frame(frame)

    process_counter += 1

    display_frame = draw_results(frame)
    display_frame = draw_status(display_frame)

    cv2.imshow("Face Recognition Attendance", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

def get_attendance_records(name_filter="", date_filter="", limit=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, name, timestamp, type
        FROM attendance
        WHERE 1 = 1
    """
    params = []

    if name_filter:
        query += " AND name LIKE ?"
        params.append(f"%{name_filter}%")

    if date_filter:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")

    query += " ORDER BY id DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_recent_attendance(limit=10):
    return get_attendance_records(limit=limit)