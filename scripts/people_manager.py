import os
import json
import unicodedata
import shutil
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
PEOPLE_JSON_PATH = os.path.join(BASE_DIR, "people.json")
DELETED_DATASET_PATH = os.path.join(BASE_DIR, "deleted_dataset")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def normalize_raw_name(text):
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )
    text = text.replace("đ", "d")

    allowed = []
    for ch in text:
        if ch.isalnum():
            allowed.append(ch)
        elif ch in [" ", "-", "_"]:
            allowed.append("_")

    raw_name = "".join(allowed)

    while "__" in raw_name:
        raw_name = raw_name.replace("__", "_")

    return raw_name.strip("_")


def ensure_dataset_dir():
    os.makedirs(DATASET_PATH, exist_ok=True)


def load_people():
    if not os.path.exists(PEOPLE_JSON_PATH):
        return {}

    try:
        with open(PEOPLE_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_people(people):
    with open(PEOPLE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(people, f, ensure_ascii=False, indent=4)


def add_or_update_person(raw_name, display_name):
    ensure_dataset_dir()

    raw_name = normalize_raw_name(raw_name)
    display_name = str(display_name or "").strip()

    if not raw_name:
        raise ValueError("Tên thư mục / mã người dùng không được để trống")

    if not display_name:
        display_name = raw_name

    person_dir = os.path.join(DATASET_PATH, raw_name)
    os.makedirs(person_dir, exist_ok=True)

    people = load_people()
    people[raw_name] = display_name
    save_people(people)

    return raw_name, display_name


def count_images(person_dir):
    if not os.path.exists(person_dir):
        return 0

    count = 0
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            count += 1

    return count


def get_people_list():
    ensure_dataset_dir()

    people_map = load_people()
    result = []

    folder_names = [
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name))
    ]

    folder_names.sort()

    for raw_name in folder_names:
        person_dir = os.path.join(DATASET_PATH, raw_name)
        display_name = people_map.get(raw_name, raw_name)
        image_count = count_images(person_dir)

        result.append({
            "raw_name": raw_name,
            "display_name": display_name,
            "image_count": image_count
        })

    return result

def get_person_dir(raw_name):
    ensure_dataset_dir()

    raw_name = normalize_raw_name(raw_name)

    if not raw_name:
        raise ValueError("Tên thư mục / mã người dùng không được để trống")

    person_dir = os.path.join(DATASET_PATH, raw_name)
    os.makedirs(person_dir, exist_ok=True)

    return person_dir


def get_person_image_count(raw_name):
    person_dir = get_person_dir(raw_name)
    return count_images(person_dir)


def get_next_image_path(raw_name):
    person_dir = get_person_dir(raw_name)

    index = 1

    while True:
        filename = f"{index:03d}.jpg"
        file_path = os.path.join(person_dir, filename)

        if not os.path.exists(file_path):
            return file_path

        index += 1

def delete_person(raw_name, backup=True):
    ensure_dataset_dir()

    raw_name = normalize_raw_name(raw_name)

    if not raw_name:
        raise ValueError("Tên thư mục / mã người dùng không được để trống")

    people = load_people()
    display_name = people.get(raw_name, raw_name)

    person_dir = os.path.join(DATASET_PATH, raw_name)
    person_exists = os.path.exists(person_dir)
    mapping_exists = raw_name in people

    if not person_exists and not mapping_exists:
        raise ValueError(f"Không tìm thấy người dùng: {raw_name}")

    backup_path = None

    if person_exists:
        if backup:
            os.makedirs(DELETED_DATASET_PATH, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder_name = f"{raw_name}_{timestamp}"
            backup_path = os.path.join(DELETED_DATASET_PATH, backup_folder_name)

            shutil.move(person_dir, backup_path)
        else:
            shutil.rmtree(person_dir)

    if mapping_exists:
        people.pop(raw_name, None)
        save_people(people)

    return {
        "raw_name": raw_name,
        "display_name": display_name,
        "backup_path": backup_path
    }