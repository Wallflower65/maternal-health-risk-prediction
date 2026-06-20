import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Generate realistic synthetic data based on UCI Maternal Health Risk dataset structure
age = np.random.randint(15, 70, n)
systolic_bp = np.random.randint(70, 180, n)
diastolic_bp = np.random.randint(50, 120, n)
bs = np.round(np.random.uniform(6, 19, n), 1)  # blood sugar
body_temp = np.round(np.random.uniform(98, 103, n), 1)
heart_rate = np.random.randint(60, 100, n)

def risk_level(row):
    score = 0
    if row['Age'] > 35 or row['Age'] < 18: score += 1
    if row['SystolicBP'] > 140: score += 2
    if row['DiastolicBP'] > 90: score += 1
    if row['BS'] > 11: score += 2
    if row['BodyTemp'] > 100: score += 1
    if row['HeartRate'] > 90: score += 1
    if score >= 4: return 'high risk'
    elif score >= 2: return 'mid risk'
    else: return 'low risk'

df = pd.DataFrame({
    'Age': age,
    'SystolicBP': systolic_bp,
    'DiastolicBP': diastolic_bp,
    'BS': bs,
    'BodyTemp': body_temp,
    'HeartRate': heart_rate
})

df['RiskLevel'] = df.apply(risk_level, axis=1)
df.to_csv('maternal_risk.csv', index=False)
print(df['RiskLevel'].value_counts())
print(df.head())
