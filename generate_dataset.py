import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

# Base features
age = np.random.randint(20, 80, n)
gender = np.random.choice(['Male', 'Female'], n)
bmi = np.random.uniform(18, 40, n)
bp = np.random.randint(90, 180, n)
glucose = np.random.randint(70, 250, n)
chol = np.random.randint(100, 350, n)
enzymes = np.random.uniform(20, 120, n)
creatinine = np.random.uniform(0.5, 3.5, n)
hemoglobin = np.random.uniform(8, 17, n)
oxygen = np.random.uniform(85, 100, n)
crp = np.random.uniform(0.1, 20, n)
smoking = np.random.randint(0, 2, n)
alcohol = np.random.randint(0, 2, n)
exercise = np.random.randint(0, 5, n)  # times/week
family = np.random.randint(0, 2, n)

# Disease logic
disease = []
for i in range(n):
    if bp[i] > 150 and chol[i] > 250:
        disease.append("Heart Disease")
    elif glucose[i] > 180:
        disease.append("Diabetes")
    elif enzymes[i] > 90:
        disease.append("Liver Disease")
    elif creatinine[i] > 2.0:
        disease.append("Kidney Disease")
    elif oxygen[i] < 90 and smoking[i] == 1:
        disease.append("Lung Disease")
    elif hemoglobin[i] < 10:
        disease.append("Anemia")
    elif bmi[i] > 35:
        disease.append("Obesity")
    elif bp[i] > 160:
        disease.append("Hypertension")
    elif crp[i] > 10 and exercise[i] < 2:
        disease.append("Stroke")
    elif family[i] == 1 and chol[i] > 230:
        disease.append("Breast Cancer")
    else:
        disease.append("Healthy")

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "BloodPressure": bp,
    "Glucose": glucose,
    "Cholesterol": chol,
    "LiverEnzyme": enzymes,
    "Creatinine": creatinine,
    "Hemoglobin": hemoglobin,
    "OxygenLevel": oxygen,
    "CRP": crp,
    "Smoking": smoking,
    "Alcohol": alcohol,
    "ExercisePerWeek": exercise,
    "FamilyHistory": family,
    "Disease": disease
})

df.to_csv("multi_disease_10.csv", index=False)
print("Dataset saved as multi_disease_10.csv")
print(df['Disease'].value_counts())
