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
short_description: Predicts the house price.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

A **machine learning project** to predict house prices in Bangalore, India, based on key features such as location, size, number of bedrooms, and bathrooms. 
The goal is to help users estimate property prices and understand the factors influencing housing costs.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)

## Project Overview

The Bangalore House Price Prediction project is an end-to-end machine learning pipeline that includes:

- Data preprocessing
- Feature engineering
- Model training and evaluation
- Web-based interface for users to input property details and get price predictions

This project aims to make house price prediction accessible and intuitive for both real estate professionals and general users.

## Features

- Predict house prices based on user input features:
  - Location
  - Square footage
  - Number of bedrooms
  - Number of bathrooms
- Uses a trained machine learning model for predictions
- Simple and interactive web interface (optional)

## Technologies Used

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `flask` / `fastapi` (for web server, if applicable)
  - `pickle` (for model serialization)
- Gradio or Streamlit (optional, for UI)

## Dataset

- The dataset contains historical housing data from Bangalore.
- Features include location, square footage, number of bedrooms (BHK), bathrooms, and price.
- The dataset is cleaned, preprocessed, and used to train the predictive model.
