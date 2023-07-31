import numpy as np
import pandas as pd

data = pd.read_csv('D:\\Kadir\\Codes\\seseker\\3-tahmin-multiple linear regression\\odev_tenis.csv')

le_data = data.iloc[:,-2:]

# preprocess
from sklearn import preprocessing
le = le_data.apply(preprocessing.LabelEncoder().fit_transform)

le_datas = data.apply(preprocessing.LabelEncoder().fit_transform)

ohe_data = le_datas.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
ohe_data = ohe.fit_transform(ohe_data).toarray()

weather = pd.DataFrame(data=ohe_data, index=range(14), columns=["o","r","s"])
final_data = pd.concat([le,weather,data.iloc[:,1:3]], axis=1)

variables = final_data.iloc[:,:-1]
to_predict = final_data.iloc[:,-1:]

# model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(variables, to_predict, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(f"Predicted: {y_pred}")

# backward elimination 1
import statsmodels.api as sm
final_data2 = np.append(arr = np.ones((14,1)).astype(int), values=variables, axis=1)
final_data2_l = variables.values
r_ols = sm.OLS(endog=to_predict, exog=final_data2_l)
r = r_ols.fit()
print(r.summary())

# backward elimination 2
variables2 = variables.iloc[:,1:]

final_data2 = np.append(arr = np.ones((14,1)).astype(int), values=variables2, axis=1)
final_data2_l = variables2.values
r_ols = sm.OLS(endog=to_predict, exog=final_data2_l)
r = r_ols.fit()
print(r.summary())

x_train1 = x_train.iloc[:,1:]
x_test1 = x_test.iloc[:,1:]
lr.fit(x_train1,y_train)
y_pred1 = lr.predict(x_test1)

# backward elimination 3
variables3_0 = variables2.iloc[:,0:3]
variables3_1 = variables2.iloc[:,-1:]
variables3 = pd.concat([variables3_0,variables3_1], axis=1)

final_data2 = np.append(arr = np.ones((14,1)).astype(int), values=variables3, axis=1)
final_data2_l = variables3.values
r_ols = sm.OLS(endog=to_predict, exog=final_data2_l)
r = r_ols.fit()
print(r.summary())

x_train1 = x_train.iloc[:,1:]
x_test1 = x_test.iloc[:,1:]
lr.fit(x_train1,y_train)
y_pred1 = lr.predict(x_test1)