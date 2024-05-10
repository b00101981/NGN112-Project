import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Customer-Lifetime-Value-Prediction.csv", header=0)
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size'] 
for c in catFeats:
	data[c] = data[c].astype('category').cat.codes

sns.histplot(data['CLV'], bins=55)
plt.title('Customer Lifetime Value (CLV) Histogram')
plt.xlabel('Customer Lifetime Value (CLV)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(20, 10))
plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Data Boxplot')
data.boxplot()

sns.pairplot(data)
plt.show()

corrMatrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# '''