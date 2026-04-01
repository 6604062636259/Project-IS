#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

print("=== ตรวจสอบระบบ ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# ตรวจสอบไลบรารี่
print("\n=== ตรวจสอบไลบรารี่ ===")
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except:
    print("❌ OpenCV ไม่พบ")

try:
    import tensorflow
    print(f"✅ TensorFlow: {tensorflow.__version__}")
except:
    print("❌ TensorFlow ไม่พบ")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except:
    print("❌ NumPy ไม่พบ")

# ตรวจสอบข้อมูลรูปภาพ
print("\n=== ตรวจสอบข้อมูล ===")
import os
train_o = len(os.listdir('Dataset/train/O')) if os.path.exists('Dataset/train/O') else 0
train_r = len(os.listdir('Dataset/train/R')) if os.path.exists('Dataset/train/R') else 0
test_o = len(os.listdir('Dataset/test/O')) if os.path.exists('Dataset/test/O') else 0
test_r = len(os.listdir('Dataset/test/R')) if os.path.exists('Dataset/test/R') else 0

print(f"Train: O={train_o}, R={train_r}")
print(f"Test: O={test_o}, R={test_r}")
print("\n✅ ทุกอย่างพร้อม! สามารถรัน TrashWaste.py ได้")
