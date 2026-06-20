# training a quick Random Forest classifier on the maternal risk data
# goal here is mainly to see how well basic vitals can predict risk level

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('maternal_risk.csv')

# sklearn needs numeric labels, not strings
le = LabelEncoder()
df['RiskLevelEncoded'] = le.fit_transform(df['RiskLevel'])

X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevelEncoded']

# 80/20 split, stratify so each risk class is represented properly in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# used class_weight='balanced' here because high risk cases dominate the data
# and i didn't want the model just defaulting to predicting high risk every time
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=le.classes_))

# confusion matrix - want to see where the model actually gets confused
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Maternal Health Risk Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# which features actually matter most to the model
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(7, 5))
importances.plot(kind='barh', color='#2E4057')
plt.title('Feature Importance - Maternal Health Risk Model')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()

print()
print("Feature importances (highest to lowest):")
print(importances.sort_values(ascending=False))

# blood sugar and systolic BP coming out on top makes sense clinically -
# matches what we learned about gestational diabetes / preeclampsia risk factors
