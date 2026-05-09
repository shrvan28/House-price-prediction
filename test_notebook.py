import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv("Housing - Housing.csv")
df
df.head()
df.tail()
df.shape
df.ndim
df.info()
df.describe()
df.index
df.dtypes
df.isnull().sum()
df.dropna()
df.fillna(0)
df.drop_duplicates()
df.drop('Address',axis=1,inplace=True)
x = df.drop('Price',axis = 1)
y = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Performance")
print("R2 Score :",r2)
print("MSE      :",mse)
print("\nEnter the house details to predict price:\n")

income = float(input("Enter Avg. Area Income: "))
age = float(input("Enter Avg. Area House Age: "))
rooms = float(input("Enter Avg. Area Number of Rooms: "))
bedrooms = float(input("Enter Avg. Area Number of Bedrooms: "))
population = float(input("Enter Area Population: "))

new_house = pd.DataFrame([[income, age, rooms, bedrooms, population]],
                         columns=x.columns)

predicted_price = model.predict(new_house)

print("\n Predicted House Price:")
print(round(predicted_price[0], 2))
import pickle

pickle.dump(model, open('model.pkl', 'wb'))

