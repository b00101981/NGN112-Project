import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

catCols = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0)
for col in catCols:
  data[col] = data[col].astype('category').cat.codes

print(data.head())

indCols = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Income', 'Loc_Code', 'M_Status', 'M_Prem', 'Mo_Claim', 'Mo_Policy', 'N_Complaints', 'N_Policies', 'P_Type', 'S_Channel', 'T_Claims', 'V_Class', 'V_Size']
depCols = ['CLV']
X = data[indCols].to_numpy().reshape(-1, len(indCols)) # Uses a random 15% of the data to test.
y = data[depCols].to_numpy()

testSize = 0.3 # Uses a random 15% of the data to test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state = 1) 

regressor = LinearRegression().fit(X_train, y_train)
y_predictions = regressor.predict(X_test)

print("Coefficients:", np.round(regressor.coef_, 5))
print("Intercept:", np.round(regressor.intercept_, 5))
print("R^2:", np.round(r2_score(y_test, y_predictions), 4))
print("Means Squared Error:", np.round(mean_squared_error(y_test, y_predictions), 4))