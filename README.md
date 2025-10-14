# Car Price Prediction — Machine Learning Project

## Overview
This repository builds a regression model to predict the **selling price of used cars** based on technical and categorical attributes.  
It includes data cleaning, encoding, feature scaling, correlation analysis, and initial machine learning models (KNN and Linear Regression).  
Model optimization and additional testing will be completed in the upcoming days.

---

## Repository Structure
"
│── notebooks/  
│   ├── data_cleaning_ML.ipynb — Data cleaning, encoding, and preprocessing  
│   ├── Model_creation_v1.ipynb — Model training and evaluation (KNN, Linear Regression)  
│  
│── data/  
│   ├── raw_data_csv/ — Original dataset  
│   ├── clean_data_csv/ — Cleaned dataset  
│  
│── config.yaml — File paths configuration  
│── README.md — Project documentation  "

---

## Data Inputs
- **Dataset source:** [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)  
- **Main features:**  
  `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`  
- **Target variable:** `selling_price`  

All paths are configured in `config.yaml`.

---

## Feature Engineering
- Extracted car brand from the `name` column.  
- Applied One-Hot Encoding to `fuel`, `seller_type`, and `transmission`.  
- Applied Ordinal Encoding to `owner`.  
- Created new feature `car_age = 2025 - year`.  
- Scaled numeric features using `StandardScaler`.  
- Checked feature correlation to support model selection.

---

## Models
- **K-Nearest Neighbors (KNN)** Regressor  
- **Linear Regression**

Each model was trained and evaluated on scaled features to compare performance (R² score, MAE, RMSE).

---

## Useful Links  

- [Kanban Board (Trello)]


- [Google Slides Presentation]


---
