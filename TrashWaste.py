import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sys # นำเข้าเพิ่มสำหรับใช้หยุดโปรแกรม

print("\n--- 0. อ่านข้อมูลภาพเพื่อเตรียมสอน (Data Loading) ---")
IMG_SIZE = 120 # กำหนดขนาดภาพ (120x120 พิกเซล)
CATEGORIES = ["O", "R"] # โฟลเดอร์ย่อย: O (Organic), R (Recyclable)

def load_images_from_folder(base_folder):
    """ฟังก์ชันสำหรับดึงภาพจากโฟลเดอร์ที่ระบุ (เช่น Dataset/train หรือ Dataset/test)"""
    image_data = []
    labels = []
    
    print(f"กำลังค้นหาในโฟลเดอร์ {base_folder} ...")
    
    # เปลี่ยนรูปเป็น Numpy Array ตรงนี้เลยครับ
    if not os.path.exists(base_folder):
        return np.array([]), np.array([])
        
    for category in CATEGORIES:
        path = os.path.join(base_folder, category)
        if not os.path.exists(path):
            continue
            
        class_num = CATEGORIES.index(category)  # O จะได้เป็นเลข 0, R จะเป็นเลข 1
            
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                
                if img_array is not None:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    image_data.append(new_array)
                    labels.append(class_num)
            except Exception as e:
                pass
                
    # แปลงเป็น Numpy Array พร้อมทำ Normalization ให้พิกเซลเป็นเลข 0-1 โดยการหาร 255.0
    return np.array(image_data) / 255.0, np.array(labels)

# 1. โหลดข้อมูลภาพโดยแยกอ่านตามโฟลเดอร์ของคุณ
print(">> โหลดภาพชุดสอน (Train)...")
X_train, y_train = load_images_from_folder("Dataset/train")

print(">> โหลดภาพชุดข้อสอบ (Test)...")
X_test, y_test = load_images_from_folder("Dataset/test")

# เช็คว่ามีรูปภาพหรือไม่ ถ้าไม่มีให้หยุดทำงานและแจ้งเตือน
if len(X_train) == 0 or len(X_test) == 0:
    print("\n[ข้อผิดพลาด] ย้ายรูปมาไม่ครบ! (หาโฟลเดอร์ train หรือ test ไม่เจอ)")
    print("--> กรุณานำโฟลเดอร์ train และ test (ที่มี O กับ R ข้างใน) ไปใส่ในโฟลเดอร์หลัก [Project IS/Dataset/]")
    sys.exit() # หยุดการทำงานของโปรแกรม

print("- - - - - - - - ")
print(f"เตรียมรูปภาพสำหรับ Train สำเร็จ: {len(X_train)} รูป")
print(f"เตรียมรูปภาพสำหรับ Test สำเร็จ: {len(X_test)} รูป")

print("\n--- 1. เริ่มขั้นตอนการสร้างโมเดล Neural Network (CNN) ---")

# 3. ออกแบบโครงสร้างโมเดล Neural Network (CNN) แบบกำหนดเอง
model_nn = Sequential([
    # เลเยอร์ที่ 1: ดึงจุดเด่นของรูปภาพ (Feature Extraction)
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    # เลเยอร์ที่ 2: ดึงจุดเด่นระดับลึกขึ้น
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # เลเยอร์ที่ 3: ตีแผ่ข้อมูลและตัดสินใจ (Classification)
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), # ป้องกันโมเดลจำข้อสอบ (Overfitting)
    Dense(1, activation='sigmoid') # ผลลัพธ์ 1 ค่า (เข้าใกล้ 0 คือ O, เข้าใกล้ 1 คือ R)
])

# 4. ตั้งค่าการเรียนรู้ของโมเดล
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.summary() # แสดงโครงสร้างโมเดล

# 5. เริ่มฝึกสอนโมเดล (Training)
# หมายเหตุ: ลองเทรนแค่ 5 รอบ (epochs=5) ก่อนเพื่อให้รันเสร็จไวๆ ถ้าอยากให้แม่นขึ้นค่อยเพิ่มทีหลังครับ
print("\nกำลังเริ่มฝึกสอนโมเดล (อาจใช้เวลาสักครู่)...")
history = model_nn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print("\nฝึกสอนโมเดล Neural Network เสร็จสิ้น! ")
model_nn.save('waste_model.h5')
print("บันทึกโมเดลสำเร็จเป็นไฟล์ 'waste_model.h5'")