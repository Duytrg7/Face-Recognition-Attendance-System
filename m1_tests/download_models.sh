#!/usr/bin/env bash
# Tải YuNet + SFace (OpenCV Zoo) vào models/. Chạy trên Pi (cần Internet).
set -e
cd "$(dirname "$0")/models"
BASE=https://github.com/opencv/opencv_zoo/raw/main/models
wget -nc "$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx"
wget -nc "$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ls -la
# Kiểm tra: file thật phải ~230 KB (YuNet) và ~37 MB (SFace). Nếu chỉ vài trăm byte là tải hỏng.
