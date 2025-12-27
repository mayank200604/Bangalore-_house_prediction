---
title: Bangalore House Prediction
emoji: 🏢
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Predicts the house price in Bangalore using Machine Learning.
---

# Bangalore House Price Prediction

This project is an end-to-end Machine Learning application that predicts house prices in Bangalore, India. It goes through the entire data science pipeline—from data cleaning and feature engineering to model building and deployment via a Flask server and a responsive web interface.

## � Live Demo

Check out the live deployment here: **[Bangalore House Prediction App](https://bangalore-house-prediction-3rvr.onrender.com)**

> **Note**: The application is deployed on a free tier instance on Render. Please wait approximately **50 seconds** for the server to spin up and load the initial request.

## �🏗 Project Structure

The project is organized into three main components:

- **`model/`**: Contains the machine learning logic (Jupyter notebook and Python script) for data processing, model training, and artifact generation.
- **`server/`**: A Flask backend that serves the trained model and handles API requests.
- **`client/`**: The frontend user interface (HTML/CSS/JS) identifying user inputs and displaying predictions.

## 🚀 Machine Learning Pipeline

The core logic resides in `model/main.py`. Here is a detailed breakdown of the steps taken:

### 1. Data Cleaning & Preprocessing
- **Data Loading**: loaded the `Bengaluru_House_Data.csv` dataset.
- **Handling Null Values**: Dropped missing values for critical features like location, size, and bathrooms.
- **Unit Conversion**:
  - `BHK`: Extracted the number of bedrooms from the 'size' column.
  - `total_sqft`: Standardized the square footage column. Ranges (e.g., "1000-1200") were converted to their average, and invalid entries were removed.
  - Filled missing `total_sqft` values with the median.

### 2. Feature Engineering & Outlier Removal
- **Outlier Detection**:
  - Removed properties where strict logic didn't apply (e.g., `total_sqft` per bedroom < 300).
  - Removed properties with an excessive number of bathrooms compared to bedrooms (`bath > bhk + 2`).
- **Dimensionality Reduction**:
  - Grouped locations with fewer than 10 data points into a single category labeled `'Other'` to reduce the number of features for One-Hot Encoding.

### 3. Model Building
- **Target Transformation**: Applied `np.log1p` (log transformation) to the price column to normalize the distribution of the target variable.
- **Encoding**: Used `pd.get_dummies` (One-Hot Encoding) for the `location` column.
- **Algorithm**: Used **LightGBM Regressor (`LGBMRegressor`)**, a high-performance gradient boosting framework.
- **Hyperparameter Tuning**: Employed `GridSearchCV` to find the best parameters (e.g., `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`).
- **Pipeline**: Encapsulated the model within a scikit-learn `Pipeline`.

### 4. Evaluation & Artifacts
- Calculated the **R² Score** on the test set to evaluate performance.
- Saved the best model as `bangalore_home_prediction_model.pickle`.
- Saved column names to `model_features.json` to ensure the server correctly maps inputs to features.

## 💻 Application Architecture

### Backend (`server/`)
- Built with **Flask**.
- **`/predict`**: POST endpoint that accepts `total_sqft`, `bhk`, `bath`, and `location`. It loads the saved model and features to return a price prediction in Lakhs.
- **`/locations`**: GET endpoint returning the list of locations for the frontend dropdown.
- Uses `util.py` to handle artifact loading and prediction logic.

### Frontend (`client/`)
- **HTML/CSS**: A clean, responsive user interface.
- **JavaScript**: Fetches locations dynamically on page load and sends asynchronous requests to the Flask backend for predictions.

## 🛠 Installation & Usage

### Prerequisites
- Python 3.x
- pip

### Steps

1. **Clone the repository** (if applicable) or navigate to the project folder.

2. **Install Dependencies**:
   Navigate to the `server` directory and install the required packages.
   ```bash
   cd server
   pip install -r requirements.txt
   ```
   *Note: Ensure `lightgbm`, `pandas`, `numpy`, `scikit-learn`, and `flask` are installed.*

3. **Run the Server**:
   ```bash
   python server.py
   ```
   The Flask server will start, typically at `http://127.0.0.1:5000`.

4. **Launch the Application**:
   Open a browser and go to `http://127.0.0.1:5000`. The server is configured to serve the static `client/index.html` at the root URL.

## 📊 Results

The model leverages gradient boosting to effectively capture non-linear relationships in real estate data. By preprocessing outliers and normalizing the target variable, the system achieves robust predictions tailored to the Bangalore market.
