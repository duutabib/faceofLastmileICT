import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


# we want to reproduce the model in labFit
dataPath = "/Users/inclusive/creativeEnergy-/LabFit/training_data_dc.csv"
data = pd.read_csv(dataPath)
print(data.columns)
print(data.dtypes)

# labFit constants
A=0.12611
B=1.0134
C=0.0000051371
D=0.009961

#Y=A/X2+B*EXP(C*X1)+D/X2**2
#Y=A*X1**(B*X2**C)+D/X2

def labfit_function(row):
    return  -1 * A * row['Static_Pa'] ** (B * row['Differential_Pa']**C) + D / row['Differential_Pa']


labfit_model = lambda row: -1*A*row.Static_Pa**(B*row.Differential_Pa**C) + D/row.Differential_Pa
labfit_model0 = lambda row: -1*A*row.Differential_Pa**(B*row.Static_Pa**C) + D/row.Static_Pa

data['labfit'] = data.apply(labfit_function, axis=1)

print(data.columns)

