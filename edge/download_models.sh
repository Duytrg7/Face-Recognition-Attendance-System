#!/usr/bin/env bash
# Tải YuNet + SFace (OpenCV Zoo) vào models/. MiniFASNet (ONNX) đã có sẵn trong repo.
# Nếu đã tải ở M1 thì chỉ cần copy: cp ../m1_tests/models/face_*.onnx models/
set -e
cd "$(dirname "$0")/models"
BASE=https://github.com/opencv/opencv_zoo/raw/main/models
wget -nc "$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx"
wget -nc "$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ls -la
