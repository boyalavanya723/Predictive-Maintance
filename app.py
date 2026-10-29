from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load("predictive_maintenance_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        machine_type = request.form["machine_type"]
        air_temp = float(request.form["air_temp"])
        process_temp = float(request.form["process_temp"])
        rotational_speed = float(request.form["rotational_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])

        machine = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        # Encode machine type
        machine = pd.get_dummies(
            machine, columns=["Type"], dtype=int
        )

        # Match the model's training columns
        machine = machine.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        result = model.predict(machine)[0]

        if result == 1:
            prediction = "Warning: Possible machine failure!"
        else:
            prediction = "The model predicts no machine failure."

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=False)