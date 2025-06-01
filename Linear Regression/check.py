# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 18:30:33 2025

@author: advit
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')


# Load your normalized files
train = pd.read_csv("train_normalised2.csv")
test = pd.read_csv("test_normalised2.csv")

# Select features and target, matching your C++ code
X_train = train.iloc[:, :-1]  # all columns except last (target)
y_train = train.iloc[:, -1]   # last column

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
# Train linear regression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on test set (normalized)
y_pred_norm = reg.predict(X_test)

# Use the same target mean and std you used in C++
mean_target = 2.07194694e+05
std_target = 1.15619125e+05

# Un-normalize predictions and targets
y_pred = y_pred_norm * std_target + mean_target
y_true = y_test * std_target + mean_target

# Calculate RMSE in original units
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"scikit-learn LinearRegression RMSE: {rmse:.2f}")

# Optionally, print a few predictions vs ground truth
for yp, yt in zip(y_pred[:5], y_true[:5]):
    print(f"Pred: {yp:.2f}   Actual: {yt:.2f}")