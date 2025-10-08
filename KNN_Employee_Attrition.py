import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ===== Step 1: Dataset =====
data = {
    "Age": [29, 35, 40, 28, 45, 25, 50, 30, 37, 26],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Sales Executive",
                "Manager", "Research Scientist", "Manager", "Sales Executive", "Laboratory Technician", "Research Scientist"],
    "MonthlyIncome": [4800, 6000, 3400, 4300, 11000, 3500, 12000, 5000, 3100, 4500],
    "JobSatisfaction": [3, 4, 2, 3, 4, 1, 4, 2, 2, 3],
    "YearsAtCompany": [4, 8, 6, 3, 15, 2, 20, 5, 9, 2],
    "Attrition": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# ===== Step 2: Encode categorical feature =====
le = LabelEncoder()
df["JobRole"] = le.fit_transform(df["JobRole"])  # encode JobRole as numbers

# Features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== Step 3: Train KNN model =====
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

print("✅ Employee Attrition Prediction Model is Ready!")

# ===== Step 4: Take user input =====
age = int(input("Enter Age: "))
job_role = input("Enter Job Role (e.g., Manager, Sales Executive, Research Scientist, Laboratory Technician): ")
monthly_income = float(input("Enter Monthly Income: "))
job_satisfaction = int(input("Enter Job Satisfaction (1-4): "))
years_at_company = int(input("Enter Years at Company: "))

# Encode job role using the same LabelEncoder
if job_role not in le.classes_:
    print("⚠️ Warning: Job Role not seen in training data. Prediction may be less accurate.")
job_role_encoded = le.transform([job_role]) if job_role in le.classes_ else [0]

# Prepare input data
employee_data = [[age, job_role_encoded[0], monthly_income, job_satisfaction, years_at_company]]

# Scale input using the same scaler
employee_data_scaled = scaler.transform(employee_data)

# ===== Step 5: Prediction =====
prediction = knn.predict(employee_data_scaled)[0]

if prediction == 1:
    print("\n🔴 Prediction: Employee is LIKELY to leave (Attrition = Yes).")
else:
    print("\n🟢 Prediction: Employee is LIKELY to stay (Attrition = No).")