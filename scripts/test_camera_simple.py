import cv2
import time

CAMERA_PATH = "/dev/video0"

cap = cv2.VideoCapture(CAMERA_PATH, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Không mở được camera")
    exit()

print("Camera opened. Đang đọc frame...")

for i in range(30):
    ret, frame = cap.read()
    print(i, ret)

    if ret:
        print("Frame shape:", frame.shape)
        cv2.imshow("Test Camera", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
