# Self-Contained Mini Project: Bank Fraud Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generate Simulated Data
np.random.seed(42)
n_samples = 1000

# Features
amount = np.random.exponential(scale=100, size=n_samples)           # Transaction amount
time = np.random.randint(0, 24, n_samples)                           # Transaction hour
oldbalance = np.random.normal(1000, 300, n_samples)                  # Account balance before transaction
newbalance = oldbalance - amount                                      # Account balance after transaction

# Fraud label (1 = fraud, 0 = non-fraud)
fraud = np.random.binomial(1, p=0.05, size=n_samples)  # 5% fraud

# Create DataFrame
df = pd.DataFrame({
    'Amount': amount,
    'Hour': time,
    'OldBalance': oldbalance,
    'NewBalance': newbalance,
    'Fraud': fraud
})

# 2. Quick Overview
print(df.head())
print("\nFraud Distribution:\n", df['Fraud'].value_counts())

# 3. Visualize Fraud Distribution
sns.countplot(x='Fraud', data=df)
plt.title('Fraud vs Non-Fraud')
plt.show()

# 4. Prepare Data
X = df.drop('Fraud', axis=1)
y = df['Fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluate Model
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature Importance (based on correlation)
corr = df.corr()['Fraud'].sort_values(ascending=False)
print("\nTop features correlated with Fraud:\n", corr)
