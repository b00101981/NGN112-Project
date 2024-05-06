import pandas as pd
import datetime

data = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv', header=0) # Importing data from csv for speed. 
catFeats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea']

# Adjusting dates then splitting into day, month, and year columns. 
i = 0
dayList = []
monthList = []
yearList = []
for d in data['Date']:
    D = str.split(d, '/')
    dayList.append(D[0])
    monthList.append(D[1])
    yearList.append(D[2])
    data.at[i, 'Date'] = datetime.date(int(D[2]), int(D[1]), int(D[0]))
    i += 1
data.insert(8, 'Day', dayList)
data.insert(8, 'Month', monthList)
data.insert(8, 'Year', yearList)

streetList = []
for a in data['Address']:
    A = str.split(a, ' ')
    streetList.append(str.join('', A[1:]))
data.insert(2, 'Street', streetList)

data.to_csv('Melbourne House Prices.csv')