from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model + vectorizer together
data = joblib.load("model_bundle.pkl")
vectorizer = data['vectorizer']
model = data['model']

topic_mapping = {
    0: 'Bank account services',
    1: 'Credit card / Prepaid card',
    2: 'Mortgages/loans',
    3: 'Theft/Dispute reporting',
    4: 'Others'
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            X = vectorizer.transform([text])
            pred_class = model.predict(X)[0]     # numeric label
            prediction = topic_mapping[pred_class]   # mapped label
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
