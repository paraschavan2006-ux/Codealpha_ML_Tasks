# CodeAlpha - Task 4: Disease Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset (Medical Data)
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 30, 55, 38, 48, 29],
    'BP': [120, 140, 130, 150, 110, 135, 160, 125, 155, 132, 145, 122],
    'Cholesterol': [200, 240, 220, 280, 180, 230, 300, 210, 290, 225, 250, 195],
    'Glucose': [90, 110, 100, 130, 85, 105, 140, 95, 135, 102, 115, 88],
    'Disease': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df[['Age', 'BP', 'Cholesterol', 'Glucose']]
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))
