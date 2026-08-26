# CodeAlpha - Task 1: Credit Scoring Model
# Author: Paras Chavan

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset (Tuzya kade dataset asel tar to load kar)
# Format: Age, Income, Credit_History, Loan_Amount, Approved

data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 30, 35, 48],
    'Income': [50000, 80000, 60000, 120000, 30000, 90000, 150000, 55000, 65000, 100000],
    'Credit_Score': [600, 700, 650, 800, 550, 720, 850, 620, 680, 750],
    'Loan_Amount': [200000, 500000, 300000, 800000, 100000, 400000, 900000, 250000, 350000, 600000],
    'Approved': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1] # 0=No, 1=Yes
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'Credit_Score', 'Loan_Amount']]
y = df['Approved']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Credit Scoring Model Trained Successfully!")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test for new customer
new_customer = [[32, 70000, 680, 300000]] # Age, Income, Credit_Score, Loan_Amount
result = model.predict(new_customer)
print(f"\nNew Customer Loan Status: {'Approved' if result[0]==1 else 'Rejected'}")
