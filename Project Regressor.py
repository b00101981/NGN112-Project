import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

sns.set_theme(rc={'figure.figsize':(10, 8.5)})

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data from csv for speed. 
catCols = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
data[catCols] = preprocessing.OrdinalEncoder().fit_transform(data[catCols]) # Encoding categorical data with sklearn; much faster during model fitting/prediction.

featureList = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Income', 'Loc_Code', 'M_Status', 'M_Prem', 'Mo_Claim', 'Mo_Policy', 'N_Complaints', 'N_Policies', 'P_Type', 'S_Channel', 'T_Claims', 'V_Class', 'V_Size']
target = 'CLV'
X = data[featureList].to_numpy().reshape(-1, len(featureList)) # Seperating and reshaping the independant (feature) columns into rows
y = data[target].to_numpy()

results = pd.DataFrame(columns=['randState', 'normMode', 'regType', 'score', 'error'])
errors = pd.DataFrame()

for randState in [1, 20, 40]:
  #print("Random State:", randState)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randState)

  for normMode in ['Z-Score/Standard', 'None', 'Min-Max']:
    #print("Normalization Mode:", normMode)
    if normMode == 'Z-Score/Standard':
      scaler = preprocessing.StandardScaler().fit(X=X_train)
      X_train = scaler.transform(X_train)
    elif normMode == 'Min-Max':
      scaler = preprocessing.MinMaxScaler().fit(X=X_train)
      X_train = scaler.transform(X_train)

    for regType in ['Linear', 'Linear SVM', 'Polynomial SVM', 'RBF SVM', 'ANN']:
      #print(regType)
      if regType == 'Linear':
        regressor = LinearRegression().fit(X_train, y_train)
      elif regType == 'Linear SVM':
        regressor = SVR(kernel = 'linear').fit(X_train, y_train) # Runs for SO LONG!!
      elif regType == 'Polynomial SVM':
        regressor = SVR(kernel = 'poly').fit(X_train, y_train) # Should degree be adjusted? Default 3. 
      elif regType == 'RBF SVM':
        regressor = SVR(kernel = 'rbf').fit(X_train, y_train)
      elif regType == 'ANN':
        regressor = MLPRegressor().fit(X_train, y_train)

      y_pred = regressor.predict(X_test)
      results.loc[len(results.index)] = [randState, normMode, regType, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)]
print(results)

#'''