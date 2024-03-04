import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

donnees = pd.read_csv('insurance.csv')

# Encodage de la variable 'sex'
donnees['sex'] = donnees['sex'].map({'female': 0, 'male': 1})

# Sélection et préparation des données
X = donnees[['age', 'sex']]
y = donnees['charges']

# Séparation des données entre hommes et femmes pour la visualisation
donnees_hommes = donnees[donnees['sex'] == 1]
donnees_femmes = donnees[donnees['sex'] == 0]

# Entraînement du modèle
modele = LinearRegression().fit(X, y)

# Prédiction
y_pred = modele.predict(X)

# Visualisation
plt.figure(figsize=(10, 6))

plt.scatter(donnees_hommes['age'], donnees_hommes['charges'], color='red', label='Hommes', alpha=0.5)
plt.scatter(donnees_femmes['age'], donnees_femmes['charges'], color='blue', label='Femmes', alpha=0.5)

# Séparation des prédictions entre hommes et femmes
y_pred_hommes = modele.predict(donnees_hommes[['age', 'sex']])
y_pred_femmes = modele.predict(donnees_femmes[['age', 'sex']])

# Tri des données pour le tracé de la ligne
indices_sorted_hommes = donnees_hommes['age'].argsort()
indices_sorted_femmes = donnees_femmes['age'].argsort()

plt.plot(donnees_hommes['age'].iloc[indices_sorted_hommes], y_pred_hommes[indices_sorted_hommes], color='violet', label='Ligne Hommes')
plt.plot(donnees_femmes['age'].iloc[indices_sorted_femmes], y_pred_femmes[indices_sorted_femmes], color='green', label='Ligne Femmes')

plt.title("Charges d'assurance par âge et sexe")
plt.xlabel('Âge')
plt.ylabel('Charges d\'assurance')
plt.legend()
plt.show()