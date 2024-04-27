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

sns.set_theme(rc={'figure.figsize':(10, 8.5)}) # Rescaling graphs is even less necessary, but helps with readability.

#print(data.head())

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data
catCols = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
data[catCols] = preprocessing.OrdinalEncoder().fit_transform(data[catCols]) # Encoding categorical data

featureList = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Income', 'Loc_Code', 'M_Status', 'M_Prem', 'Mo_Claim', 'Mo_Policy', 'N_Complaints', 'N_Policies', 'P_Type', 'S_Channel', 'T_Claims', 'V_Class', 'V_Size']
target = 'CLV'
X = data[featureList].to_numpy().reshape(-1, len(featureList)) # Seperating and reshaping the independant (feature) columns into rows 
y = data[target].to_numpy()

for randState in [1, 20, 40]:
  print("Random State:", randState)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randState) 

  for normMode in ['None', 'Z-Score/Standard', 'Min-Max']:
    print("Normalization Mode:", normMode)
    if normMode == 'Z-Score/Standard':
      scaler = preprocessing.StandardScaler().fit(x=X_train, y=y_train)
      X_train = scaler.transform(X_train)
    elif normMode == 'Min-Max':
      scaler = preprocessing.MinMaxScaler().fit(x=X_train, y=y_train)
      X_train = scaler.transform(X_train)

    for regressorType in ['Linear', 'Linear SVM', 'Polynomial SVM', 'RBF SVM', 'ANN']:
      print(regressorType)
      if regressorType == 'Linear':
        regressor = LinearRegression().fit(X_train, y_train)
      elif regressorType == 'Linear SVM':
        regressor = SVR(kernel = 'linear').fit(X_train, y_train) # Runs prepetually fsr
      elif regressorType == 'Polynomial SVM':
        regressor = SVR(kernel = 'poly').fit(X_train, y_train)
      elif regressorType == 'RBF SVM':
        regressor = SVR(kernel = 'rbf').fit(X_train, y_train)
      elif regressorType == 'ANN':
        regressor = MLPRegressor().fit(X_train, y_train)

      y_pred = regressor.predict(X_test)
      print("R^2:", np.round(r2_score(y_test, y_pred), 4))
      print("Means Squared Error:", np.round(mean_squared_error(y_test, y_pred), 4))
      print('\n')
    print('\n\n')
  print('\n\n\n')
  


#print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test), np.shape(y_pred))

#print("Coefficients:", np.round(regressor.coef_, 5))
#print("Intercept:", np.round(regressor.intercept_, 5))
#print("R^2:", np.round(r2_score(y_test, y_pred), 4))
#print("Means Squared Error:", np.round(mean_squared_error(y_test, y_pred), 4))

# Setting up the heatmap
#corrMatrix = data.copy().corr()
#heatMap = sns.heatmap(data=corrMatrix, annot=True)
#plt.xticks(range(0, len(featureList+depCols)), featureList+depCols, rotation=90)
#plt.yticks(range(0, len(featureList+depCols)), featureList+depCols)
#plt.show()