import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

donnees = pd.read_csv('insurance.csv')

# Encodage des variables catégorielles 'sex' et 'smoker' en valeurs numériques
donnees['sex'] = donnees['sex'].map({'female': 0, 'male': 1})
donnees['smoker'] = donnees['smoker'].map({'no': 0, 'yes': 1})

# Sélection et préparation des données
X = donnees[['age', 'smoker']]
y = donnees['charges']

# Séparation des données entre hommes et femmes pour entraîner des modèles distincts
donnees_hommes = donnees[donnees['sex'] == 1]
donnees_femmes = donnees[donnees['sex'] == 0]

# Entraînement des modèles
modele_hommes = LinearRegression().fit(donnees_hommes[['age', 'smoker']], donnees_hommes['charges'])
modele_femmes = LinearRegression().fit(donnees_femmes[['age', 'smoker']], donnees_femmes['charges'])

# Prédiction
donnees_hommes['prediction'] = modele_hommes.predict(donnees_hommes[['age', 'smoker']])
donnees_femmes['prediction'] = modele_femmes.predict(donnees_femmes[['age', 'smoker']])

# Visualisation
plt.figure(figsize=(10, 6))

plt.scatter(donnees_hommes['age'], donnees_hommes['charges'], color='red', label='Hommes', alpha=0.5)
plt.scatter(donnees_femmes['age'], donnees_femmes['charges'], color='blue', label='Femmes', alpha=0.5)

# Tri des données pour le tracé de la ligne
indices_sorted_hommes = donnees_hommes['age'].argsort()
indices_sorted_femmes = donnees_femmes['age'].argsort()

plt.plot(donnees_hommes['age'].iloc[indices_sorted_hommes], donnees_hommes['prediction'].iloc[indices_sorted_hommes], color='violet', label='Ligne Hommes')
plt.plot(donnees_femmes['age'].iloc[indices_sorted_femmes], donnees_femmes['prediction'].iloc[indices_sorted_femmes], color='green', label='Ligne Femmes')

plt.title("Charges d'assurance par âge, sexe et statut de fumeur")
plt.xlabel('Âge')
plt.ylabel('Charges d\'assurance')
plt.legend()
plt.show()
