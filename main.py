import pandas as pd

print("Predictive Maintenance Project Started!")

data = pd.read_csv("data/raw/ai4i2020.csv")

print("\nDataset Preview:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)
print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nDataset Information:")
data.info()
print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nMachine Failure Counts:")
print(data["Machine failure"].value_counts())

print("\nFailure Percentage:")
print(data["Machine failure"].value_counts(normalize=True) * 100)
# Prepare data for machine learning

# Remove columns that are not useful for prediction
X = data.drop(
    columns=["UDI", "Product ID", "Machine failure",
             "TWF", "HDF", "PWF", "OSF", "RNF"]
 
)
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Target: whether the machine failed
y = data["Machine failure"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Results")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predict machine failures
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Predict machine failures
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
import joblib

# Save the trained model
joblib.dump(model, "predictive_maintenance_model.pkl")

print("\nModel saved successfully!")