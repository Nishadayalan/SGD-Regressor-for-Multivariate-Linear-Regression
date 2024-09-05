# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End

## Program:


Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
\nDeveloped by: NISHA.D 
\nRegisterNumber: 212223230143



```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data=fetch_california_housing()
print(data)
```
## Output:
![image](https://github.com/user-attachments/assets/f2d81866-25e8-43d0-b2d1-f9cd51283d5d)
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
## Output:
![image](https://github.com/user-attachments/assets/f1e66f98-aec8-4a92-be2b-78fef6e74e58)
```
df.info()
```
## Output:
![image](https://github.com/user-attachments/assets/d5398fb7-94ad-406b-a73e-65c297334621)
```
x=df.drop(columns=['AveOccup','target'])
x.info()
```
## Output:
![image](https://github.com/user-attachments/assets/59d07200-b3b0-4a7f-8040-3fb2b49158ec)
```
y=df[['AveOccup','target']]
y.info()
```
## Output:
![image](https://github.com/user-attachments/assets/5600ec74-45b8-42fc-9358-105044807ddd)
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x.head()
```
## Output:
![image](https://github.com/user-attachments/assets/02122ea4-4471-4247-88cf-1ef18e237e77)
```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)
```
## Output:
![image](https://github.com/user-attachments/assets/d5390dba-6662-435c-8a78-9bafbb5e0522)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
```
## Output:
![image](https://github.com/user-attachments/assets/45bef840-3f14-4c0a-b9bf-c502a4809ef1)
```
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
```
## Output:
![image](https://github.com/user-attachments/assets/490eee2d-01b3-4d05-9d98-03af74336697)
```
print("\nPredictions:\n", y_pred[:5])
```
## Output:

![image](https://github.com/user-attachments/assets/e8bc552e-8bbc-451b-a8ff-667089e65389)
## Result:

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
