
# Loan Default Risk Prediction System

## Project Overview

This project is a Machine Learning-based Loan Default Risk Prediction System developed using Python and Streamlit.

The application predicts the probability of a loan applicant defaulting on a loan and classifies applicants into risk categories.

## Machine Learning Model

- Logistic Regression
- Random Forest (model experimentation and comparison)
- One-Hot Encoding
- Standard Scaling
- Class balancing
- ROC-AUC evaluation
- Threshold optimization

## Application Features

- Individual applicant prediction
- Bulk Excel file upload
- Batch predictions for multiple applicants
- Default probability calculation
- Risk classification
- Applicant review recommendations
- Prediction history
- Interactive dashboard
- Risk distribution charts
- CSV downloads for all, high-risk, and low-risk applicants

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Dataset

The project uses a loan default dataset containing 10,000 applicant records and 14 input features.

Target variable:
- 0 = No Default
- 1 = Default

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the Streamlit application:

   streamlit run app.py

## Project Objective

To demonstrate how Machine Learning can support loan risk assessment by identifying potentially high-risk applicants and assisting manual credit review.

## Disclaimer

This project is developed for educational and portfolio demonstration purposes. Predictions should not be used as the sole basis for actual lending decisions.
