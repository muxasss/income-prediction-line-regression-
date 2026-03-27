from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='tempates')


model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = float(data["age"])
    experience = float(data["experience"])

    X = np.array([[age, experience]])

    prediction = model.predict(X)[0]

    return jsonify({
        "income": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)

