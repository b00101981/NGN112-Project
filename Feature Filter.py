import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Customer-Lifetime-Value-Prediction.csv', header=0) # Importing data from csv for speed. 
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']

for c in catFeats:
    data[c] = data[c].astype('category').cat.codes

corrMatrix = data.corr()
sigFeats = {}
i = 0
for c in corrMatrix['CLV']:
    if abs(c) > 0.03:
        sigFeats[corrMatrix.index[i]] = c
    i += 1

print(sigFeats) #Ignore CLV
plt.figure(figsize=(20, 20))
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()