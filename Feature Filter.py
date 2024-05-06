import pandas as pd

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data from csv for speed. 
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']
numFeats = ['Income', 'M_Prem', 'Mo_Claim', 'Mo_Policy', 'N_Complaints', 'N_Policies', 'T_Claims']

for c in catFeats:
    data[c] = data[c].astype('category').cat.codes
corrMatrix = data.corr(method='pearson')
gFeatDict = {}
i = 0
for c in corrMatrix['CLV']:
    if abs(c) > 0.025:
        gFeatDict[corrMatrix.index[i]] = c
    i += 1

print(gFeatDict) #Ignore CLV