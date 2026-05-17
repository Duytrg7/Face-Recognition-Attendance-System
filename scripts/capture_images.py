import cv2
import os

PERSON_NAME = "nguyen_van_a"  # đổi tên này trước khi chạy
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset')
SAVE_PATH = os.path.join(DATASET_PATH, PERSON_NAME)

os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"Chụp ảnh cho: {PERSON_NAME}")
print("Nhấn SPACE để chụp, Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Đã chụp ảnh {count}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Hoàn thành! Đã chụp {count} ảnh")