import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- 1. โหลดข้อมูล Dataset ---")
try:
    df = pd.read_csv('Employee Churn.csv')
    print("โหลดไฟล์สำเร็จ! ข้อมูล 5 บรรทัดแรก:\n", df.head())
    print("\nเช็คจำนวนค่าว่าง (Missing Values) ในแต่ละคอลัมน์ก่อนคลีน:")
    print(df.isnull().sum())
except FileNotFoundError:
    print("❌ ไม่พบไฟล์ 'Employee Churn.csv' กรุณาตรวจสอบชื่อไฟล์และตำแหน่งที่อยู่ครับ")
    exit()

print("\n--- 2. ขั้นตอนการเตรียมข้อมูล (Data Preprocessing) ---")
# 2.1 จัดการค่าว่าง (Missing Values)
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())
if 'MonthlyIncome' in df.columns:
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
if 'Satisfaction_Score' in df.columns:
    df['Satisfaction_Score'] = df['Satisfaction_Score'].fillna(df['Satisfaction_Score'].mode()[0])

# 2.1.2 จัดการข้อมูลผิดปกติ (Outliers)
if 'Age' in df.columns:
    df.loc[df['Age'] > 100, 'Age'] = df['Age'].median() # แก้ไขอายุ 150 ที่เป็นค่าผิดปกติ
if 'YearsAtCompany' in df.columns:
    df['YearsAtCompany'] = df['YearsAtCompany'].apply(lambda x: abs(x) if x < 0 else x) # แก้ค่าอายุงานที่ติดลบ
if 'Satisfaction_Score' in df.columns:
    df.loc[df['Satisfaction_Score'] > 5, 'Satisfaction_Score'] = 5.0 # คะแนนไม่ควรเกิน 5

# 2.2 จัดการชื่อแผนกให้เป็นมาตรฐานเดียวกัน (แก้ปัญหา IT, I.T., Sales, sales)
if 'Department' in df.columns:
    df['Department'] = df['Department'].str.lower().str.replace('.', '', regex=False).str.replace('sales', 'sale').str.replace('sale', 'sales')
    df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# 2.3 จัดการคอลัมน์เป้าหมาย (Target)
target_column = 'Attrition'
if target_column in df.columns:
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
else:
    print(f"\n⚠️ ไม่พบคอลัมน์ '{target_column}' กรุณาตรวจสอบชื่อคอลัมน์ในไฟล์ CSV ครับ")
    exit()

# ลบคอลัมน์ ID เพราะไม่มีผลต่อการทำนาย
if 'Employee_ID' in df.columns:
    df = df.drop('Employee_ID', axis=1)

print("\nข้อมูลหลังคลีนพร้อมเทรนโมเดล:\n", df.head(3))

print("\n--- 3. เริ่มสร้างโมเดล Ensemble (อย่างน้อย 3 ประเภท) ---")
# แบ่งข้อมูล X และ y
X = df.drop(target_column, axis=1)
y = df[target_column]

# ปรับสเกลข้อมูลให้สมดุลสำหรับ SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# กำหนดโมเดล 3 ประเภทตามข้อกำหนด
model1 = RandomForestClassifier(n_estimators=50, random_state=42)
model2 = SVC(probability=True, random_state=42)
model3 = GradientBoostingClassifier(n_estimators=50, random_state=42)

# รวมเป็น Ensemble แบบ Voting
ensemble_model = VotingClassifier(
    estimators=[('rf', model1), ('svm', model2), ('gb', model3)],
    voting='soft'
)

print("\nกำลังฝึกสอนโมเดล Ensemble...")
ensemble_model.fit(X_train, y_train)

# ทดสอบและประเมินผล
y_pred = ensemble_model.predict(X_test)
print("\n--- 4. ผลลัพธ์การทำงานของโมเดล Ensemble ---")
print(f"ความแม่นยำ (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nรายละเอียด Classification Report:\n", classification_report(y_test, y_pred))

# 5. บันทึกโมเดลและ Scaler สำหรับนำไปใช้บนหน้าเว็บ
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nบันทึกโมเดลสำเร็จ! ได้ไฟล์ 'ensemble_model.pkl' และ 'scaler.pkl'")