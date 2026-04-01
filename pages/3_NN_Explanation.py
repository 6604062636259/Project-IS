import streamlit as st

st.set_page_config(page_title="NN Explanation", page_icon="🧠")

st.title("🧠 Neural Network: วิเคราะห์รูปภาพขยะ")
st.markdown("---")

st.header("1. ข้อมูลรูปภาพขยะ (Image Dataset)")
st.write("เราใช้ข้อมูลรูปภาพแบ่งเป็น 2 หมวดหมู่ O และ R:")
st.markdown("""
- **O (Organic):** ขยะอินทรีย์ที่ย่อยสลายได้ เช่น เศษอาหาร ผลไม้ ใบไม้
- **R (Recyclable):** ขยะที่นำมารีไซเคิลได้ เช่น ขวดพลาสติก กระดาษ แก้วน้ำ
""")

st.header("2. การเตรียมรูปภาพ (Preprocessing)")
st.write("ก่อนให้ให้ AI เรียนรู้รูปภาพ ต้องทำการแปลงภาพให้อยู่ในฟอร์แมตมาตรฐานซะก่อนโดย:")
st.info("""
1. แปลง **สีของภาพ (BGR to RGB)** เพราะ OpenCV อ่านเป็น BGR ซึ่งไม่สอดคล้องกับธรรมชาติ
2. ย่อขนาดภาพเป็น **120x120 pixels** เพื่อลดการคำนวณที่หนักหน่วง
3. ปรับค่าความสว่างให้อยู่ในช่วง **0-1 (Normalization)** โดยการนำค่าพิกเซลไปหาร 255.0
""")

st.header("3. โครงสร้างโมเดล (Convolutional Neural Network)")
st.write("เนื่องจากโมเดลเป็นการมองภาพเลยต้องใช้ CNN ที่เลียนแบบการมองเห็นของสมองคนประกอบไปด้วย:")
st.markdown("""
- **1st Conv2D+MaxPooling (32 filters):** สำหรับการจับขอบ รูปร่าง และเส้นตรงพื้นฐานที่เล็กๆ ในภาพ
- **2nd Conv2D+MaxPooling (64 filters):** สร้างฟีเจอร์รูปร่างที่ซับซ้อนขึ้นอย่างมุม หรือทรงกลม เช่น ทรงของขวดน้ำ
- **Flatten & Dense (128 units):** เปลี่ยนแผ่นภาพที่วิเคราะห์ได้ให้กลายเป็นแถวข้อมูล 1 มิติเพื่อให้ระบบตัดสินใจ
- **Dropout (0.5):** ปิดการทำงาน 50% แบบสุ่มเพื่อป้องกัน AI 'จำข้อสอบ (Overfitting)'
- **Output Layer (Sigmoid):** ใช้ 1 Node ในการตัดสินผลลัพธ์:
  - ค่าที่ใกล้ **0** คือ ขยะอินทรีย์ (Organic - O)
  - ค่าที่ใกล้ **1** คือ ขยะรีไซเคิล (Recyclable - R)
""")

st.success("**ข้อสรุป:** การใช้ CNN ดึงจุดเด่นของภาพ ช่วยให้ไม่ต้องหาหลักเกณฑ์การแยกขยะเองด้วยมือ ปล่อยให้ระบบ Deep Learning จัดการให้แม่นยำที่สุด")
st.markdown("---")
st.caption("อ้างอิงชุดข้อมูล (Dataset): [Waste Classification Data จาก Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)")
st.caption("ทฤษฎี: Deep Learning (Convolutional Neural Network) โดยใช้ TensorFlow กบ OpenCV")
