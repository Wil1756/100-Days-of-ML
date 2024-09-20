import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

df =pd.read_csv('winequality-white.csv',sep=';')
print(df.head())
df.dropna()
x= df.loc[:,df.columns!='quality']
y= df['quality']


lr_model = LinearRegression().fit(x,y)
print(lr_model.score(x,y))