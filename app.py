from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from scipy import stats

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

input_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'city', 'statezip']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-house-price", methods=["POST"])
def predict():
    if request.form['country'] != 'USA':
        return render_template("index.html", prediction="This model is only trained on USA data. Please enter a USA address.")

    input_data = {
        'bedrooms': float(request.form["bedrooms"]),
        'bathrooms': float(request.form["bathrooms"]),
        'sqft_living': float(request.form["sqft_living"]),
        'sqft_lot': float(request.form["sqft_lot"]),
        'floors': float(request.form["floors"]),
        'waterfront': float(request.form["waterfront"]),
        'view': float(request.form["view"]),
        'condition': float(request.form["condition"]),
        'sqft_above': float(request.form["sqft_above"]),
        'sqft_basement': float(request.form["sqft_basement"]),
        'yr_built': float(request.form["yr_built"]),
        'yr_renovated': float(request.form["yr_renovated"]),
        'city': request.form["city"],
        'statezip': request.form["statezip"],
        'street': request.form["street"],
        'country': request.form["country"],
        'date': request.form["date"]
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.drop(['country', 'street', 'date'], axis=1)

    try:
        input_df['city'] = encoder.transform(input_df[['city']])
    except ValueError:
        input_df['city'] = encoder.transform([['unknown']])[0]

    try:
        input_df['statezip']= input_df['statezip'].str.extract(r'(\d+)').astype(int)
    except ValueError:
        input_df['statezip'] = encoder.transform([['unknown']])[0]

    input_df[input_cols] = scaler.transform(input_df[input_cols])

    prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=f"{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)