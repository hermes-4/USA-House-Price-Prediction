
## Project Overview

The project includes a Jupyter Notebook for data exploration, preprocessing, and model training, as well as a Flask web application for predicting house prices based on user input.

### Key Features

- **Date:** Sale date of the property.
- **Price:** Target variable (sale price in USD).
- **Bedrooms:** Number of bedrooms.
- **Bathrooms:** Number of bathrooms.
- **Sqft Living:** Living area size (sqft).
- **Sqft Lot:** Lot size (sqft).
- **Floors:** Number of floors.
- **Waterfront:** Waterfront view (1 = yes, 0 = no).
- **View:** View quality (0-4).
- **Condition:** Property condition (1-5).
- **Sqft Above:** Above-ground area (sqft).
- **Sqft Basement:** Basement area (sqft).
- **Yr Built:** Year built.
- **Yr Renovated:** Year last renovated.
- **Street, City, Statezip, Country:** Location details.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kelvinmhacwilson/USA-House-Price-Prediction.git
    cd USA-House-Price-Prediction
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate 
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebook

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook Final_Project.ipynb
    ```

2. Run the cells to explore the data, preprocess it, and train the machine learning model.

### Flask Web Application

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Fill in the form with the property details and click "Predict" to get the estimated house price.

## Model

The trained model, along with the encoder and scaler, are saved in the [model](http://_vscodecontentref_/4)