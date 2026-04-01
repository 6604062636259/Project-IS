import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Testing", page_icon="🔎")
st.title("🔎 ทดสอบทายผลพนักงานลาออก (ML)")

# โหลดโมเดล
try:
    model = joblib.load('ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("✅ โหลดโมเดลพร้อมใช้งาน")
except:
    st.error("❌ ไม่พบไฟล์โมเดล กรุณากลับไปรันโปรแกรม `EmployeeDataset.py` ใหม่")
    st.stop()

st.markdown("### กรอกข้อมูลพนักงานเพื่อทดสอบ")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (Age)", min_value=18, max_value=65, value=25)
    department = st.selectbox("แผนก (Department)", options=["HR", "IT", "Sales"])
with col2:
    salary = st.number_input("เงินเดือน (Monthly Income)", min_value=9000, value=30000)
    years = st.number_input("อายุงาน (Years At Company)", min_value=0, value=2)

score = st.slider("คะแนนความพึงพอใจ (1=แย่มาก, 5=ดีมาก)", 1, 5, 3)

if st.button("🔮 เริ่มทายผลการลาออก", use_container_width=True):
    # เตรียมตารางข้อมูลตามโมเดล
    # Columns expected config (from the get_dummies structure in training):
    # [Age, MonthlyIncome, YearsAtCompany, Satisfaction_Score, Department_it, Department_sales]
    
    dept_it = 1 if department == "IT" else 0
    dept_sales = 1 if department == "Sales" else 0
    
    # สร้าง DataFrame แถวเดียว
    input_df = pd.DataFrame([{
        'Age': age,
        'MonthlyIncome': salary,
        'YearsAtCompany': years,
        'Satisfaction_Score': float(score),
        'Department_it': dept_it,
        'Department_sales': dept_sales
    }])
    
    # ปรับสเกล
    input_scaled = scaler.transform(input_df)
    
    # ทายผล
    pred = model.predict(input_scaled)[0]
    
    st.markdown("---")
    if pred == 1 or pred == "Yes" or pred == 'Yes':
        st.error("### 🚨 ผลวิเคราะห์: พนักงานรายนี้ **'มีเกณฑ์การลาออก'**")
        st.write("ควรเรียกคุยเพื่อนำเสนอแผนพัฒนาหรือฟังปัญหาที่เกิดขึ้น")
    else:
        st.success("### 💚 ผลวิเคราะห์: พนักงานรายนี้ **'พอใจที่จะอยู่ต่อ'**")
        st.write("การบริหารจัดการบุคคลอยู่ในเกณฑ์ดี")
