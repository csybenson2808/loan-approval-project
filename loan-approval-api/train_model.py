import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load data
df = pd.read_csv("loan_data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDistribution of 'approved' labels:")
print(df['approved'].value_counts())

print("\nFeature data types:")
print(df.dtypes)

features = ["age", "income", "credit_score", "loan_amount", "loan_term_months",
            "interest_rate", "num_existing_loans", "has_criminal_record", "dti"]

X = df[features]
y = df["approved"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTotal samples: {len(df)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train model with class_weight balanced
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"\nRandom Forest model accuracy: {accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC score (using predicted probabilities)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

print("\nSample predictions:", y_pred[:10])
print("Actual labels:     ", y_test.values[:10])

# Save model and feature list
joblib.dump(model, "loan_model.joblib")
joblib.dump(features, "features_list.joblib")
print("\nModel and features saved.")
