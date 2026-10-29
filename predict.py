import joblib
import pandas as pd

# Load the saved model
model = joblib.load("predictive_maintenance_model.pkl")

# Example machine readings
machine = pd.DataFrame([{
    "Type": "L",
    "Air temperature [K]": 303,
    "Process temperature [K]": 313,
    "Rotational speed [rpm]": 1300,
    "Torque [Nm]": 65,
    "Tool wear [min]": 250
}])

# Convert machine type into the same format used during training
machine = pd.get_dummies(machine, columns=["Type"], dtype=int)

# Ensure the columns match the model's training features
machine = machine.reindex(columns=model.feature_names_in_, fill_value=0)

# Make a prediction
prediction = model.predict(machine)[0]

print("Machine Failure Prediction:")

if prediction == 1:
    print("Warning: The model predicts a possible machine failure.")
else:
    print("The model predicts no machine failure.")