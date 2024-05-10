import pandas as pd
import numpy as np

data = pd.read_csv("Customer-Lifetime-Value-Prediction.csv", header=0)
catFeats = ['State', 'Coverage', 'Education', 'Emp_Status', 'Gender', 'Loc_Code', 'M_Status', 'P_Type', 'S_Channel', 'V_Class', 'V_Size']

print(data.describe())

medians = data.median(numeric_only=True)
print("The medians of numerical features are:\n", medians)

modes = pd.DataFrame()
for feat in catFeats:
	modes[feat] = data[feat].mode()
print("The modes of categorical features are:\n", modes)

print("The relative frequencies of each feature's data are:\n", data.value_counts(normalize=True))

#'''