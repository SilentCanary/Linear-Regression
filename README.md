# 🏡 LINEAR REGRESSION – Housing Price Prediction (California Housing Dataset)

This project predicts housing prices using linear regression. It is divided into two parts: **data preprocessing in Python** and **model implementation in C++**.

---

## 🔹 Part 1: Preprocessing in Python

- Analyzed the dataset using a **correlation heatmap** to find the most important features.
- Created a new feature: `bedrooms_per_room`.
- Selected only three features features:
  - `median_income`
  - `ocean_proximity`
  - `bedrooms_per_room`
- Applied **one-hot encoding** to the `ocean_proximity` column. // this is beacause prices near ocean were more expensive than inland)
- Normalized the selected features using standard scaling.
- Saved the final processed datasets as:
  - `train_normalised2.csv`
  - `test_normalised2.csv`

---

## 🔹 Part 2: Linear Regression in C++

- Implemented **Linear Regression from scratch** using **Batch Gradient Descent**.
- Implements : hypothesis calculation, cost function, and gradient computation.
- Used the processed CSV files for training and testing.
- The **RMSE (Root Mean Square Error)** on test data is close to the result from **Scikit-Learn's** Linear Regression using the same features, making it a good comparison.

---

