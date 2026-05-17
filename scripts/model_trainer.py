import os
import pickle
import cv2
import face_recognition


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_image_paths(dataset_path=DATASET_PATH):
    image_paths = []

    if not os.path.exists(dataset_path):
        return image_paths

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file))

    image_paths.sort()
    return image_paths


def train_model(progress_callback=None):
    image_paths = list_image_paths(DATASET_PATH)

    known_encodings = []
    known_names = []

    total_images = len(image_paths)
    processed_images = 0
    skipped_images = 0

    if total_images == 0:
        raise ValueError("Không tìm thấy ảnh nào trong dataset/")

    if progress_callback:
        progress_callback(f"Bắt đầu train model với {total_images} ảnh...")

    for index, image_path in enumerate(image_paths, start=1):
        processed_images += 1

        raw_name = os.path.basename(os.path.dirname(image_path))

        if progress_callback:
            progress_callback(
                f"Đang xử lý {index}/{total_images}: {raw_name}"
            )

        image = cv2.imread(image_path)

        if image is None:
            skipped_images += 1
            if progress_callback:
                progress_callback(f"Bỏ qua ảnh lỗi: {image_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=0,
            model="hog"
        )

        if len(boxes) == 0:
            boxes = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=1,
                model="hog"
            )

        encodings = face_recognition.face_encodings(
            rgb,
            boxes,
            num_jitters=1,
            model="small"
        )

        if len(encodings) == 0:
            skipped_images += 1
            if progress_callback:
                progress_callback(f"Không tạo được encoding, bỏ qua: {image_path}")
            continue

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(raw_name)

    if len(known_encodings) == 0:
        raise ValueError("Không tạo được encoding nào. Hãy kiểm tra ảnh dataset.")

    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(data))

    result = {
        "total_images": total_images,
        "processed_images": processed_images,
        "skipped_images": skipped_images,
        "total_encodings": len(known_encodings),
        "encodings_path": ENCODINGS_PATH
    }

    if progress_callback:
        progress_callback("Train model hoàn tất.")

    return result


if __name__ == "__main__":
    def print_progress(message):
        print(f"[INFO] {message}")

    result = train_model(progress_callback=print_progress)

    print("[INFO] Training complete.")
    print(f"[INFO] Total images: {result['total_images']}")
    print(f"[INFO] Total encodings: {result['total_encodings']}")
    print(f"[INFO] Skipped images: {result['skipped_images']}")
    print(f"[INFO] Saved to: {result['encodings_path']}")