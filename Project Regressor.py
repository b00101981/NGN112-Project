import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data from csv for speed. 
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
data[catFeats] = preprocessing.OrdinalEncoder().fit_transform(data[catFeats]) # Encoding categorical data with sklearn; much faster during model fitting/prediction.

sigFeats = ['Coverage', 'M_Prem', 'N_Complaints', 'T_Claims', 'V_Class'] # Determined by Feature Filter script
target = 'CLV'
X = data[sigFeats].to_numpy().reshape(-1, len(sigFeats)) # Seperating and reshaping the independant (feature) columns into rows. 
y = data[target].to_numpy()

results = pd.DataFrame(columns=['randState', 'normMode', 'regType', 'score', 'mse'])
for randState in [1, 20, 40]:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randState) # Splitting of data does not take non-signficant features in the test data. Problem?

  for normMode in ['None', 'Z-Score/Standard', 'Min-Max']: #
    if normMode == 'Z-Score/Standard': # There is no point in scaling y (or X, in fact). 
      scaler = preprocessing.StandardScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test) # Must scale test data with the mean and std of training data. 
    elif normMode == 'Min-Max':
      scaler = preprocessing.MinMaxScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

    for regType in ['Linear', 'Linear SVM', 'Polynomial SVM', 'RBF SVM', 'ANN']: 
      #Terminal warning says that not shrinking "may be faster". Not true, I checked. Epsilon is hardly making a difference, data must be bunched. 
      if regType == 'Linear':
        regressor = LinearRegression().fit(X_train, y_train) # Nothing to optimize. 
      elif regType == 'Linear SVM':
        regressor = SVR(kernel = 'linear').fit(X_train, y_train) # Runs for SO LONG! ~3-5 mil iterations. Shrinking is necessary. 
      elif regType == 'Polynomial SVM':
        regressor = SVR(kernel = 'poly', degree=2, C=6000).fit(X_train, y_train) # Higher degrees get wildly inaccurate, even by this code's standards. 
      elif regType == 'RBF SVM': # Auto gamma and no shrinking are much slower, not worth it. 
        regressor = SVR(kernel = 'rbf', C=500).fit(X_train, y_train) # swapping to gamma='auto' is sometimes better and sometimes worse. Not worth it. 
      elif regType == 'ANN': #"Warm Start" speeds it up by a lot, but yields very different results. Size taken from rule of thumb, Input*2/3+Output. 
        regressor = MLPRegressor(hidden_layer_sizes=[4,], tol=0.05, max_iter=10000000, n_iter_no_change=8, learning_rate_init=0.07).fit(X_train, y_train) # Many possible optimizations. The only solver fitting of the dataset is adam and no n_iter_no_change and tol are where they minimize overfitting. (Relative to default all numbers are out of wack)
      y_pred = regressor.predict(X_test)
      results.loc[len(results.index)] = [randState, normMode, regType, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)]
print(results)