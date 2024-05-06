import pandas as pd
import datetime

data = pd.read_csv('Melbourne House Prices.csv', header=0) # Importing data from csv for speed. 
catFeats = ['Suburb', 'Address', 'Street', 'Type', 'Method', 'SellerG', 'Date', 'Regionname', 'CouncilArea']

for c in catFeats:
    data[c] = data[c].astype('category').cat.codes
corrMatrix = data.corr(method='pearson')
gFeatDict = {}
i = 0
for c in corrMatrix['Price']:
    if abs(c) > 0.1:
        gFeatDict[corrMatrix.index[i]] = c
    i += 1

print(gFeatDict) #Ignore Price