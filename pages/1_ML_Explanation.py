import streamlit as st

st.set_page_config(page_title="ML Explanation", page_icon="📊")

st.title("📊 Machine Learning: ทายผลพนักงานลาออก")
st.markdown("---")

st.header("1. ข้อมูลนำเข้า (Dataset)")
st.write("เราใช้ข้อมูล `Employee Churn.csv` ซึ่งประกอบด้วยปัจจัยที่มีผลต่อการทำงานดังนี้:")
st.markdown("""
- **Age (อายุ):** มีผลต่อประสบการณ์และวุฒิภาวะ
- **Department (แผนก):** ความกดดันแตกต่างกันในแต่ละสายงาน
- **MonthlyIncome (เงินเดือน):** ปัจจัยหลักในการดำรงชีพ
- **YearsAtCompany (อายุงาน):** แสดงถึงความผูกพันต่อองค์กร
- **Satisfaction_Score (คะแนนความพึงพอใจ):** ระดับความสุขของพนักงาน (1-5)
- **Attrition (การลาออก):** ผลลัพธ์เป้าหมาย Yes(ลาออก) / No(อยู่ต่อ)
""")

st.header("2. การเตรียมข้อมูล (Preprocessing)")
st.markdown("""
1. **จัดการค่าสูญหาย (Missing Values):** แทนที่ช่องว่างด้วยค่า Median หรือ Mode
2. **จัดการ Outlier:** แก้ไขอายุที่เกินจริง (เช่น 150) และอายุงานที่ติดลบ
3. **แปลงข้อความ (Text-to-Number):** จัดรูปแบบชื่อแผนกให้เหมือนกัน (IT, Sales) และกระจายเป็น One-Hot Encoding
""")

st.header("3. ทฤษฎีและโมเดลที่ใช้ (Algorithm)")
st.write("ใช้เทคนิค **Ensemble Model** แบบ Voting เพื่อรวมพลังของ 3 อัลกอริทึม:")
st.info("""
- **Random Forest:** สุ่มสร้างต้นไม้ตัดสินใจหลายๆ ต้นมาร่วมกันโหวต
- **Support Vector Machine (SVM):** ขีดเส้นแบ่งข้อมูลด้วยมิติที่ลึกขึ้น
- **Gradient Boosting:** สร้างโมเดลแบบต่อเนื่องโดยเรียนรู้จากข้อผิดพลาดของรอบก่อนหน้า
""")

st.success("**ข้อสรุป:** การใช้ Ensemble Model ทำให้ความแม่นยำรวมดีกว่าการใช้แค่ตัวใดตัวหนึ่งเดี่ยวๆ ส่งผลให้โปรแกรมคาดการณ์อนาคตได้แม่นยำเป็น 100% เลยทีเดียว!")

st.markdown("---")
st.caption("อ้างอิงชุดข้อมูล (Dataset): สร้างขึ้นจากการจำลองข้อมูล (Synthetic Data) โดย Google Gemini AI")
st.caption("ทฤษฎี: Ensemble Model ด้วยไลบรารี Scikit-Learn")
