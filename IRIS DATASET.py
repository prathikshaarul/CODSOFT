# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the dataset
df = pd.read_csv("D:\Prathiksha\Programs\IRIS.csv")  # Ensure the file is in the same directory

# Display basic info
print(df.info())
print(df.head())

# Pairplot visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue="species", diag_kind="hist", palette="husl")
plt.show()

# Encode species labels
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Split data into features and target
X = df.drop(columns=["species"])
y = df["species"]

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Visualization
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()
