import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data 
insurance_data = pd.read_csv("insurance.csv")

plt.scatter(insurance_data['age'], insurance_data['charges'], color='red')
plt.title('Analyse de donnée')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Variable de la régréssion linéaire
X = insurance_data['age'].values.reshape(-1, 1)
y = insurance_data['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Régression linéaire - Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()