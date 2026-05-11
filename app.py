from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static",
)

# Load model
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load columns
df = pd.read_csv("data/cleaned_churn.csv")
FEATURES = df.drop("Churn", axis=1).columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {}

    for feature in FEATURES:
        data[feature] = 0

    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])

    contract = request.form["contract"]

    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly

    if contract == "one":
        data["Contract_One year"] = 1
    elif contract == "two":
        data["Contract_Two year"] = 1

    input_df = pd.DataFrame([data])

    prob = model.predict_proba(input_df)[0][1]

    return render_template(
        "result.html",
        probability=round(prob * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
