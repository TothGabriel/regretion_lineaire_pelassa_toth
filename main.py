import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

house_data = pd.read_csv('insurance.csv')

plt.scatter(house_data['age'], house_data['charges'], color='red')
plt.title('Analyse de donnée')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

X = house_data.iloc[:, :-1].values
y = house_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
