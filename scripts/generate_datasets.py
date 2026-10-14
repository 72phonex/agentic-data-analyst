import os
import random
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# 1. Sales Transactions Dataset
np.random.seed(42)
random.seed(42)
n_sales = 200

dates = pd.date_range("2024-01-01", periods=60, freq="D")
categories = ["Electronics", "Home & Kitchen", "Apparel", "Books", "Sports"]
regions = ["North", "South", "East", "West"]
statuses = ["Active", "Completed", "Pending", "Cancelled"]

sales_data = []
for i in range(1, n_sales + 1):
    cat = random.choice(categories)
    price_ranges = {
        "Electronics": (50.0, 800.0),
        "Home & Kitchen": (15.0, 200.0),
        "Apparel": (10.0, 120.0),
        "Books": (8.0, 45.0),
        "Sports": (20.0, 300.0)
    }
    low, high = price_ranges[cat]
    price = round(random.uniform(low, high), 2)
    qty = random.randint(1, 5)
    discount = round(random.choice([0.0, 0.05, 0.1, 0.15, 0.2]), 2)
    region = random.choice(regions)
    date = random.choice(dates).strftime("%Y-%m-%d")
    status = random.choices(["Active", "Completed", "Cancelled"], weights=[0.45, 0.40, 0.15])[0]
    cancelled_flag = 1 if status == "Cancelled" else 0
    customer_id = f"CUST_{random.randint(100, 150)}"
    sales_data.append({
        "order_id": f"ORD_{i:04d}",
        "customer_id": customer_id,
        "order_date": date,
        "category": cat,
        "unit_price": price,
        "quantity": qty,
        "discount": discount,
        "total_amount": round(price * qty * (1 - discount), 2),
        "status": status,
        "cancelled_flag": cancelled_flag,
        "region": region
    })

df_sales = pd.DataFrame(sales_data)
df_sales.to_csv("data/sales_transactions.csv", index=False)
print("Created data/sales_transactions.csv (200 rows)")

# 2. Student Performance Dataset
n_students = 150
departments = ["Computer Science", "Business", "Engineering", "Arts", "Biology"]
genders = ["Female", "Male"]

student_data = []
for i in range(1, n_students + 1):
    dept = random.choice(departments)
    gender = random.choice(genders)
    study_hours = round(max(2.0, random.gauss(15.0, 5.0)), 1)
    attendance = round(min(100.0, max(50.0, random.gauss(82.0, 10.0))), 1)
    # Midterm correlated with study hours and attendance
    midterm = round(min(100.0, max(30.0, 30 + 2.2 * study_hours + 0.3 * attendance + random.gauss(0, 5))), 1)
    final = round(min(100.0, max(30.0, 0.4 * midterm + 2.0 * study_hours + 0.2 * attendance + random.gauss(0, 6))), 1)
    total_score = round(0.4 * midterm + 0.6 * final, 1)
    pass_status = "Pass" if total_score >= 60.0 else "Fail"
    scholarship = "Yes" if total_score >= 85.0 else "No"
    student_data.append({
        "student_id": f"STU_{i:04d}",
        "gender": gender,
        "department": dept,
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "midterm_score": midterm,
        "final_score": final,
        "total_score": total_score,
        "pass_status": pass_status,
        "scholarship": scholarship
    })

df_students = pd.DataFrame(student_data)
df_students.to_csv("data/student_performance.csv", index=False)
print("Created data/student_performance.csv (150 rows)")

# 3. Community Health Survey Dataset
n_health = 180
smoker_opts = ["Non-Smoker", "Former Smoker", "Current Smoker"]
exercise_opts = ["Sedentary", "Moderate", "Active"]

health_data = []
for i in range(1, n_health + 1):
    age = random.randint(22, 75)
    gender = random.choice(["Female", "Male"])
    bmi = round(max(17.5, min(42.0, random.gauss(26.5, 4.5))), 1)
    exercise = random.choices(exercise_opts, weights=[0.35, 0.45, 0.20])[0]
    smoker = random.choices(smoker_opts, weights=[0.60, 0.25, 0.15])[0]
    # Blood pressure related to age, BMI, and smoking
    sbp = int(round(100 + 0.4 * age + 0.8 * bmi + (10 if smoker == "Current Smoker" else 0) + random.gauss(0, 8)))
    dbp = int(round(65 + 0.2 * age + 0.5 * bmi + random.gauss(0, 6)))
    cholesterol = int(round(150 + 0.8 * age + 1.2 * bmi + random.gauss(0, 15)))
    # Risk flag
    diabetes_risk = "High" if (bmi > 30.0 or (age > 50 and sbp > 140)) else "Low"
    health_data.append({
        "participant_id": f"HLT_{i:04d}",
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "exercise_frequency": exercise,
        "smoker_status": smoker,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "cholesterol": cholesterol,
        "diabetes_risk": diabetes_risk
    })

df_health = pd.DataFrame(health_data)
df_health.to_csv("data/health_survey.csv", index=False)
print("Created data/health_survey.csv (180 rows)")
