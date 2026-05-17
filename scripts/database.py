import sqlite3
import os
from datetime import datetime
import unicodedata

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def normalize_text(text):
    text = str(text or "").casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )
    text = text.replace("đ", "d")
    return text

def get_last_attendance_type(name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT type
        FROM attendance
        WHERE name = ?
        ORDER BY id DESC
        LIMIT 1
    """, (name,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return row[0]


def get_next_attendance_type(name):
    last_type = get_last_attendance_type(name)

    if last_type == "Vào":
        return "Ra"

    return "Vào"


def insert_attendance(name):
    attendance_type = get_next_attendance_type(name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attendance (name, timestamp, type)
        VALUES (?, ?, ?)
    """, (name, timestamp, attendance_type))

    conn.commit()
    conn.close()

    return attendance_type, timestamp


def get_all_attendance():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, timestamp, type
        FROM attendance
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows

def get_attendance_records(name_filter="", date_filter="", limit=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, name, timestamp, type
        FROM attendance
        WHERE 1 = 1
    """
    params = []

    if date_filter:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Lọc tên bằng Python để hỗ trợ tìm kiếm không dấu
    if name_filter:
        normalized_filter = normalize_text(name_filter)

        rows = [
            row for row in rows
            if normalized_filter in normalize_text(row[1])
        ]

    if limit is not None:
        rows = rows[:limit]

    return rows


def get_recent_attendance(limit=10):
    return get_attendance_records(limit=limit)

def clear_attendance_records():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='attendance'")

    conn.commit()
    conn.close()