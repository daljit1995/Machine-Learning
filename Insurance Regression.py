    # ---------- Insurance Data ------------

import pandas as pd
import numpy as np
import joblib

Reg=pd.read_csv('insurance.csv')
# 1. chceck Missing Value
Reg.isna().sum()

# 2. Dependent & Independent Data Store Sepratly
Reg_x=Reg.iloc[:,:-1]
Reg_y=Reg.iloc[:,-1]
print(Reg_x['gender'].value_counts()) # it is use for check element in that row

# 3. Clean data for convernt into numeric value
# gender (LabelEncoder) beacuse it have two elements
from sklearn.preprocessing import LabelEncoder
gender = LabelEncoder()
Reg_x["gender"] = gender.fit_transform(Reg_x["gender"])
joblib.dump(gender,'gender.joblib')

smoker = LabelEncoder()
Reg_x["smoker"] = smoker.fit_transform(Reg_x["smoker"])
joblib.dump(smoker,'smoker.joblib')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([('encoding',OneHotEncoder(),[5])],remainder='passthrough')
Reg_x=ct.fit_transform(Reg_x)
joblib.dump(ct,'ct.joblib')

# 4. Standard Scalling
from sklearn.preprocessing import StandardScaler as SC
sc=SC()
Reg_x=sc.fit_transform(Reg_x)
joblib.dump(sc,'sc.joblib')

# When data is long then use PCA

# 5. Train test Splitting
from sklearn.model_selection import train_test_split as tts
Reg_x_train,Reg_x_test,Reg_y_train,Reg_y_test=tts(Reg_x,Reg_y,test_size=0.2,random_state=0)

# 6. Testing
# 1. Linear Regression Testing
from sklearn.linear_model import LinearRegression as LR
regressor=LR()
regressor.fit(Reg_x_train,Reg_y_train)
joblib.dump(regressor,'regressor.joblib')

y_pred=regressor.predict(Reg_x_test)

# Find out error use this function only for regression
# Metrics
# 1.Metrics Absolute = y_pred - Rec_y_test
# from sklearn.metrics import r2_score
# print(r2_score(Rec_y_test,y_pred))

# r2_score - r2_score tells us how much variance of dependent variable is explained by independent column

# Tuning when data below 0.80 in linear regression

# 2. Support Vector Regression
from sklearn.svm import SVR
regressor=SVR(C=10000) # C= Regularization Parameter (when parameter is low then it close to )  
regressor.fit(Reg_x_train,Reg_y_train)
joblib.dump(regressor,'regressor.joblib')

y_pred=regressor.predict(Reg_x_test)

# 3. Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor as LR

regressor=LR()
regressor.fit(Reg_x_train,Reg_y_train)
joblib.dump(regressor,'regressor.joblib')

y_pred=regressor.predict(Reg_x_test)

# 4. Random Forest
from sklearn.ensemble import RandomForestRegressor as LR
regressor=LR()
regressor.fit(Reg_x_train,Reg_y_train)
joblib.dump(regressor,'regressor.joblib')

y_pred=regressor.predict(Reg_x_test)

# 7. Evaluation

from sklearn.metrics import r2_score
print(r2_score(Reg_y_test,y_pred))

print(Reg_y_test['charges'].value_counts())

print(y_pred['charges'].value_counts())
