
# Titanic Survival Prediction – Feature Engineering Assignment

## Overview
This project performs data cleaning, feature engineering, and feature selection on the Titanic dataset to prepare features for machine learning models predicting passenger survival.

Dataset: Titanic – Machine Learning from Disaster

## Project Structure

titanic_assignment/
│
├── data/
│   └── train.csv (add dataset here)
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── requirements.txt
└── README.md

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run scripts

python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py

## Key Observations

- Female passengers had higher survival rates.
- First class passengers survived more often.
- Fare and passenger class strongly influence survival.
