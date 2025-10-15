# Car Price Prediction — Machine Learning Project

## 🚗 Overview
This project builds a **regression model to predict the selling price of used cars** based on their technical specifications and categorical attributes.  
The workflow includes data cleaning, feature engineering, encoding, scaling, correlation analysis, outlier handling, and multiple machine learning algorithms.  
The main objective is to identify the model that best predicts car prices with high accuracy.

---

## 🗂️ Repository Structure

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
- **Data version:** `clean_data_v3_irma.csv` (includes PowerTransformer and outlier removal) 

All paths are configured in `config.yaml`.

---

## Feature Engineering
- Extracted car brand from the `name` column.  
- Applied One-Hot Encoding to `fuel`, `seller_type`, and `transmission`.  
- Applied Ordinal Encoding to `owner`.  
- Created new feature `car_age = 2025 - year`.  
- Scaled numerical features with **MinMaxScaler** and normalized with **PowerTransformer**.  
- Checked **feature correlations** to avoid redundancy.  
- Removed extreme outliers in the `selling_price` column after transformation.

---

## 🤖 Models and Evaluation

### Baseline Models
- **K-Nearest Neighbors (KNN)**  
- **Linear Regression**

### Ensemble and Advanced Models
- **Bagging Regressor (Decision Tree base)**  
- **Pasting Regressor (Decision Tree base)**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **AdaBoost Regressor**

Each model was evaluated using:
- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)  
- **R²** (Coefficient of Determination)

**Best performing model:**  
✅ `Random Forest Regressor` — **R² ≈ 0.93**

---

## Useful Links  

- [Kanban Board (Trello)]


- [Google Slides Presentation]


---
