import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
donnees = pd.read_csv('insurance.csv')
donnees_simplifiees = donnees[['age', 'charges']]

# Séparer les données entre hommes et femmes
donnees_hommes = donnees[donnees['sex'] == 'male']
donnees_femmes = donnees[donnees['sex'] == 'female']

# Préparation des données pour la régression
X_hommes = donnees_hommes[['age']]
y_hommes = donnees_hommes['charges']
X_femmes = donnees_femmes[['age']]
y_femmes = donnees_femmes['charges']

# Entraînement des modèles de régression linéaire
modele_hommes = LinearRegression().fit(X_hommes, y_hommes)
modele_femmes = LinearRegression().fit(X_femmes, y_femmes)
y_pred_hommes = modele_hommes.predict(X_hommes)
y_pred_femmes = modele_femmes.predict(X_femmes)

# Visualisation
plt.figure(figsize=(10, 6))

# Données et lignes de régression pour les hommes
plt.scatter(X_hommes['age'], y_hommes, color='red', label='Hommes', alpha=0.5)
indices_sorted_hommes = X_hommes['age'].argsort()
plt.plot(X_hommes['age'].iloc[indices_sorted_hommes], y_pred_hommes[indices_sorted_hommes], color='violet', label='Ligne Hommes')


# Données et lignes de régression pour les femmes
plt.scatter(X_femmes['age'], y_femmes, color='blue', label='Femmes', alpha=0.5)
indices_sorted_femmes = X_femmes['age'].argsort()
plt.plot(X_femmes['age'].iloc[indices_sorted_femmes], y_pred_femmes[indices_sorted_femmes], color='green', label='Ligne Femmes')


plt.title("Charges d'assurance par âge pour hommes et femmes")
plt.xlabel('Âge')
plt.ylabel('Charges d\'assurance')
plt.legend()
plt.show()