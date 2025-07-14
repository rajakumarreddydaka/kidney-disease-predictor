from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('kidney.pkl', 'rb'))

# Feature preprocessing mapping (as used in training)
mapping = {
    "rbc": {"abnormal": 1, "normal": 0},
    "pc": {"abnormal": 1, "normal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
}

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/kidney", methods=["GET"])
def kidneyPage():
    return render_template("kidney.html")

@app.route("/predict", methods=["POST"])
def predictPage():
    try:
        # Get form data
        form_data = request.form

        # Extract and preprocess features
        features = [
            float(form_data["age"]),
            float(form_data["bp"]),
            float(form_data["al"]),
            float(form_data["su"]),
            mapping["rbc"].get(form_data["rbc"], 0),
            mapping["pc"].get(form_data["pc"], 0),
            mapping["pcc"].get(form_data["pcc"], 0),
            mapping["ba"].get(form_data["ba"], 0),
            float(form_data["bgr"]),
            float(form_data["bu"]),
            float(form_data["sc"]),
            float(form_data["pot"]),
            float(form_data["wc"]),
            mapping["htn"].get(form_data["htn"], 0),
            mapping["dm"].get(form_data["dm"], 0),
            mapping["cad"].get(form_data["cad"], 0),
            mapping["pe"].get(form_data["pe"], 0),
            mapping["ane"].get(form_data["ane"], 0)
        ]

        # Convert to numpy array and reshape
        input_np = np.array(features).reshape(1, -1)

        # Predict
        pred = model.predict(input_np)[0]

        return render_template("predict.html", pred=pred)

    except Exception as e:
        print("Error during prediction:", e)
        return render_template("home.html", message="Please enter valid input data.")

if __name__ == "__main__":
    app.run(debug=False)
