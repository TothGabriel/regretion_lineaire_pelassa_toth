import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset 
insurance_data = pd.read_csv("insurance.csv")

# Calcul écart interquartile
Q1 = insurance_data['charges'].quantile(0.25)
Q3 = insurance_data['charges'].quantile(0.75)
IQR = Q3 - Q1

# Bornes pour les valeurs non aberrantes
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrage des données aberrantes
insurance_data_filtered = insurance_data[(insurance_data['charges'] >= lower_bound) & (insurance_data['charges'] <= upper_bound)]

plt.scatter(insurance_data_filtered['age'], insurance_data_filtered['charges'], color='red')
plt.title('Analyse de donnée après nettoyage')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

X = insurance_data_filtered['age'].values.reshape(-1, 1)
y = insurance_data_filtered['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Création et entraînement du modèle de régression linéaire
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Régression linéaire - Age vs Charges après nettoyage')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()
