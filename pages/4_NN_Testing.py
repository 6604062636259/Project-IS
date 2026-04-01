import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(page_title="NN Testing", page_icon="📸")
st.title("📸 ทดสอบวิเคราะห์รูปภาพขยะ (NN)")
st.markdown("---")

IMG_SIZE = 120

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_model.h5')

try:
    model = load_model()
    st.success("✅ โหลดโมเดล Neural Network (CNN) สำเร็จพร้อมใช้งานแล้วครับ!")
except:
    st.error("❌ ไม่พบไฟล์โมเดลรูปภาพขยะ (waste_model.h5) กรุณารัน `TrashWaste.py` เพื่อ Train Model ก่อนครับ")
    st.stop()

st.write("### อัปโหลดรูปภาพเพื่อทดสอบระบบ")
st.info("💡 ทริค: เลือกรูปภาพขยะที่เห็นชัดเจน เช่น ขวดพลาสติก(R) หรือ แอปเปิ้ลเน่า(O)")

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพขยะ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # อ่านและแสดงรูป
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="รูปภาพที่อัปโหลด", width=300)
    
    if st.button("🔍 เริ่มวิเคราะห์ประเภทขยะ!", use_container_width=True):
        st.write("กำลังประมวลผลผ่านระบบ Convolutional Neural Network...")
        
        # Preprocessing รูปภาพให้เหมือนที่สอน
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0  # Normalization
        img_input = np.expand_dims(img_normalized, axis=0)  # ขยายมิติเพิ่ม Batch
        
        # คาดเดา
        prediction = model.predict(img_input)[0][0]
        
        st.markdown("---")
        if prediction >= 0.5:
            percent = prediction * 100
            st.success(f"### ♻️ เป็น **ขยะรีไซเคิล (Recyclable)**")
            st.write(f"ความมั่นใจของ AI: {percent:.2f}%")
        else:
            percent = (1 - prediction) * 100
            st.error(f"### 🍂 เป็น **ขยะอินทรีย์ (Organic)**")
            st.write(f"ความมั่นใจของ AI: {percent:.2f}%")
