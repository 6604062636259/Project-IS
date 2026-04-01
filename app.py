import streamlit as st

st.set_page_config(page_title="AI Project Hub", page_icon="🚀", layout="wide")

st.title("🚀 ศูนย์รวมโปรเจกต์ AI & Machine Learning")
st.markdown("---")

st.write("ยินดีต้อนรับสู่เว็บแอปพลิเคชันสำหรับนำเสนอและทดสอบโมเดลปัญญาประดิษฐ์")
st.write("ระบบนี้แบ่งการทำงานออกเป็น 2 โปรเจกต์หลัก ดังนี้:")

col1, col2 = st.columns(2)

with col1:
    st.info("### 1. Employee Churn Prediction")
    st.write("**เทคนิคที่ใช้:** Machine Learning (Ensemble Model)")
    st.write("**รายละเอียด:** ทายผลล่วงหน้าว่าพนักงานคนไหนในบริษัทมีโอกาสที่จะลาออก โดยวิเคราะห์จากประวัติและข้อมูลต่างๆ เช่น อายุงาน, แผนก, เงินเดือน และคะแนนความพึงพอใจ")
    st.write("👉 คุณสามารถศึกษาทฤษฎีหรือทดสอบการทำงานจริงได้ที่เมนูด้านซ้ายมือครับ")

with col2:
    st.success("### 2. Waste Image Classification")
    st.write("**เทคนิคที่ใช้:** Deep Learning (Neural Network - CNN)")
    st.write("**รายละเอียด:** ระบบ AI อัจฉริยะที่สามารถมองภาพขยะและแยกแยะได้ว่าเป็นขยะประเภทอินทรีย์ (Organic) หรือ ขยะรีไซเคิล (Recyclable) อัตโนมัติ")
    st.write("👉 กรุณาเลือกเมนูด้านซ้ายเพื่ออ่านกระบวนการสร้างและทดสอบระบบได้เลยครับ")

st.markdown("---")
st.caption("พัฒนาโดยใช้ Streamlit ร่วมกับ Scikit-Learn และ TensorFlow")
