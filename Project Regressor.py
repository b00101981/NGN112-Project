import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from cycler import cycler

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data from csv for speed.
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
data[catFeats] = preprocessing.OrdinalEncoder().fit_transform(data[catFeats]) # Encoding categorical data with sklearn; much faster during model fitting/prediction.

sigFeats = ['Coverage', 'M_Prem', 'N_Complaints', 'T_Claims', 'V_Class'] # Determined by Feature Filter script
X = data[sigFeats].to_numpy().reshape(-1, len(sigFeats)) # Seperating and reshaping the independant (feature) columns into rows.
y = data['CLV'].to_numpy()

colors = ['#696969', '#a9a9a9', '#556b2f', '#8b4513', '#228b22', '#8b0000', '#808000', '#483d8b', '#3cb371', '#b8860b', '#008b8b', '#4682b4', '#000080', '#d2691e', '#9acd32', '#cd5c5c', '#32cd32', '#8fbc8f', '#8b008b', '#b03060', '#d2b48c', '#ff0000', '#ffa500', '#ffd700', '#0000cd', '#00ff00', '#9400d3', '#00fa9a', '#dc143c', '#00ffff', '#9370db', '#adff2f', '#da70d6', '#ff00ff', '#1e90ff', '#f0e68c', '#dda0dd', '#90ee90', '#ff1493', '#ffa07a', '#afeeee', '#87cefa', '#7fffd4', '#fffacd', '#ffb6c1']
plt.rcParams.update({'font.size': 5, 'axes.prop_cycle': cycler('color', colors)})
plt.figure(figsize=(20, 20))
legend = []

results = pd.DataFrame(columns=['randState', 'normMode', 'regType', 'score', 'mse', 'coefficients', 'intercepts'])
for randState in [1, 20]: #, 40
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randState)

  for normMode in ['None', 'Z-Score/Standard', 'Min-Max']: #
    if normMode == 'Z-Score/Standard': # There is no point in scaling y (or X, in fact). 
      scaler = preprocessing.StandardScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test) # Must scale test data with the mean and std of training data. 
    elif normMode == 'Min-Max':
      scaler = preprocessing.MinMaxScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

    for regType in ['Linear', 'Linear SVM', 'Polynomial SVM', 'RBF SVM', 'ANN']: #
      # Terminal warning says that not shrinking "may be faster". Not true, I checked. Epsilon is hardly making a difference, data must be bunched. 
      if regType == 'Linear':
        regressor = LinearRegression().fit(X_train, y_train) # Nothing to optimize. 
        coefList = regressor.coef_
        intList = regressor.intercept_
        linInt = regressor.intercept_

      elif regType == 'Linear SVM':
        regressor = SVR(kernel = 'linear').fit(X_train, y_train) # Runs for SO LONG! ~3-5 mil iterations. Shrinking is necessary. 
        coefList = regressor.coef_
        intList = regressor.intercept_

      elif regType == 'Polynomial SVM':
        regressor = SVR(kernel = 'poly', degree=2, C=6000).fit(X_train, y_train) # Higher degrees get wildly inaccurate, even by this code's standards. 
        coefList = None
        intList = regressor.intercept_

      elif regType == 'RBF SVM': # Auto gamma and no shrinking are much slower, not worth it. 
        regressor = SVR(kernel = 'rbf', C=500).fit(X_train, y_train) 
        coefList = None
        intList = regressor.intercept_

      elif regType == 'ANN': #"Warm Start" speeds it up by a lot, but yields very different results. 
        regressor = MLPRegressor(random_state=randState, hidden_layer_sizes=[4], tol=0.06, max_iter=15000, n_iter_no_change=8, learning_rate_init=0.05).fit(X_train, y_train) # Many possible optimizations. The only solver fitting of the dataset is adam and no n_iter_no_change and tol are where they minimize overfitting. (Relative to default all numbers are out of wack)
        coefList = regressor.coefs_
        intList = regressor.intercepts_

      y_pred = regressor.predict(X_test)
      results.loc[len(results.index)] = [randState, normMode, regType, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), coefList, intList]

      legend.append(str(randState)+' '+normMode+' '+regType)
      plt.plot(X_test, y_pred, '.', linewidth=0)
      

print(results[['randState', 'normMode', 'regType', 'score', 'mse']])
results.to_csv('Regression Results.csv')

plt.legend(legend+['True'])
plt.plot(X_test, y_test, '.', linewidth=0)
plt.show()

# '''