import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('Melbourne House Prices.csv', header=0).dropna() # Importing data from csv for speed. 
catFeats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea']
data[catFeats] = preprocessing.OrdinalEncoder().fit_transform(data[catFeats]) # Encoding categorical data with sklearn; much faster during model fitting/prediction.

sigFeats = ['Suburb', 'Rooms', 'Type', 'Distance'] # Determined by Feature Filter script
target = 'Price'
X = data[sigFeats].to_numpy().reshape(-1, len(sigFeats)) # Seperating and reshaping the independant (feature) columns into rows. 
y = data[target].to_numpy()

results = pd.DataFrame(columns=['randState', 'normMode', 'regType', 'score', 'mse', 'coefficients', 'intercepts'])
for randState in [1]: #, 20, 40
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randState) # Splitting of data does not take non-signficant features in the test data. Problem?

  for normMode in ['None', 'Z-Score/Standard']: #, 'Min-Max'
    if normMode == 'Z-Score/Standard': # There is no point in scaling y (or X, in fact). 
      scaler = preprocessing.StandardScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test) # Must scale test data with the mean and std of training data. 
    elif normMode == 'Min-Max':
      scaler = preprocessing.MinMaxScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

    for regType in ['Linear', 'Polynomial SVM', 'RBF SVM']: #, 'Linear SVM', 'ANN'
      #Terminal warning says that not shrinking "may be faster". Not true, I checked. Epsilon is hardly making a difference, data must be bunched. 
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
        regressor = SVR(kernel = 'poly', degree=3, coef0=linInt).fit(X_train, y_train) # Higher degrees get wildly inaccurate, even by this code's standards. 
        #polyCoefs = preprocessing.PolynomialFeatures(degree=2).fit(X_train) 
        #coefList = pd.DataFrame(polyCoefs.transform(X_train).tolist(), columns=list(polyCoefs.powers_.reshape(5, 21)))
        coefList = None
        intList = regressor.intercept_
      
      elif regType == 'RBF SVM': # Auto gamma and no shrinking are much slower, not worth it. 
        regressor = SVR(kernel = 'rbf', epsilon=3.5).fit(X_train, y_train) # swapping to gamma='auto' is sometimes better and sometimes worse. Not worth it. 
        coefList = None
        intList = regressor.intercept_

      elif regType == 'ANN': #"Warm Start" speeds it up by a lot, but yields very different results. Size taken from rule of thumb, Input*2/3+Output. 
        regressor = MLPRegressor(random_state=randState, hidden_layer_sizes=[4, 20], tol=0.06, max_iter=10000000, n_iter_no_change=8, learning_rate_init=0.005).fit(X_train, y_train) # Many possible optimizations. The only solver fitting of the dataset is adam and no n_iter_no_change and tol are where they minimize overfitting. (Relative to default all numbers are out of wack)
        coefList = regressor.coefs_
        intList = regressor.intercepts_

      y_pred = regressor.predict(X_test)
      results.loc[len(results.index)] = [randState, normMode, regType, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), coefList, intList]
#print(results[['randState', 'normMode', 'regType', 'score', 'mse']])
#results.to_csv('Regression Results.csv')

# '''