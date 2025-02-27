import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("D:\Prathiksha\Programs\creditcard.csv")

# Separate features and target variable
X = df.drop(columns=["Class"])
y = df["Class"]

# Normalize "Amount" and "Time" features
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance using undersampling
fraud_indices = y_train[y_train == 1].index
genuine_indices = y_train[y_train == 0].index
undersample_genuine_indices = np.random.choice(genuine_indices, size=len(fraud_indices), replace=False)
undersample_indices = np.concatenate([fraud_indices, undersample_genuine_indices])
X_train, y_train = X_train.loc[undersample_indices], y_train.loc[undersample_indices]

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
