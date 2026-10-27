# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

nbpix = 32

# ------------ Téléchargement des données ------------

train_dir = r"C:\Users\23mat\OneDrive\Bureau\TIPE\CONCOURS TIPE\data_set_dog_cat\archive\PetImages"
categories = ["Cat", "Dog"]

features = []
targets = []

for label, category in enumerate(categories):
    path = os.path.join(train_dir, category, "*.jpg")
    for file in glob.glob(path):
        try:
            img = Image.open(file).resize((nbpix, nbpix)).convert('L')
            features.append(np.array(img).flatten() / 255.0)  # Normalisation
            if label == 0:
                targets.append(np.array([1, 0]))
            else:
                targets.append(np.array([0, 1]))
        except:
            print(f"/!\\ Problème avec {file} /!\\")

# Prendre un subset si nécessaire
features_ = np.array(features[20000:])
targets_ = np.array(targets[20000:])
indices = np.random.permutation(len(features_))
I_test_data = features_[indices]
Y_test_data = targets_[indices]

Data_set = [(I_test_data[i], Y_test_data[i]) for i in range(len(I_test_data))]
print(f"✅ {len(Data_set)} images test chargées")

# ------------ Chargement des poids et bias ------------

nompoids1 = input('Nom du fichier des poids 1 (sans le txt) : ')
W1 = np.loadtxt(nompoids1 + '.txt')  # W1.shape = (784, hidden_size)

nombias1 = input('Nom du fichier des bias 1 (sans le txt) : ')
bias1 = np.loadtxt(nombias1 + '.txt') 

nompoids2 = input('Nom du fichier des poids 2 (sans le txt) : ')
W2 = np.loadtxt(nompoids2 + '.txt')  # W2.shape = (hidden_size, 10)

nombias2 = input('Nom du fichier des bias 2 (sans le txt) : ')
bias2 = np.loadtxt(nombias2 + '.txt')


# ------------ test image en particulier ------------

# def calque(I, W1, b1, W2, b2):
#     # Première couche (activation sigmoïde)
#     Z1 = I.dot(W1) + b1
#     A1 = 1 / (1 + np.exp(-Z1))  # sigmoïde

#     # Deuxième couche (sortie softmax)
#     Z2 = A1.dot(W2) + b2
#     expZ = np.exp(Z2 - np.max(Z2))
#     A2 = expZ / np.sum(expZ)
#     return np.argmax(A2)


def calque(I, W1, b1, W2, b2, activation="softmax"):
    """
    Calcule la sortie du réseau à deux couches :
    - Première couche : sigmoïde
    - Deuxième couche : sigmoïde
    """
    # /!\ Il s'agit de la version sigmoïde
    
    # Aplatit l'image si elle est 2D (28x28)
    if I.ndim == 2:
       I = I.flatten()

    # --- Couche 1 ---
    Z1 = I.dot(W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))  # sigmoïde

    # --- Couche 2 ---
    Z2 = A1.dot(W2) + b2

    A2 = 1 / (1 + np.exp(-Z2))
    return np.argmax(A2)  # pour rester comparable


# ------------ Efficacité ------------

Nb_Erreur = 0

for i in range(len(I_test_data)):
    
    if calque(I_test_data[i], W1, bias1, W2, bias2) != np.argmax(Y_test_data[i]):
        Nb_Erreur+=1

print("L'algorithme est précis à la hauteur de :", 100*(len(I_test_data)-Nb_Erreur)/len(I_test_data),"%")


# def precision(I_test_data, Y_test_data, W1, b1, W2, b2, activation="sigmoide"):
#     bonnes = 0
#     for i in range(len(I_test_data)):
#         pred = calque(I_test_data[i], W1, b1, W2, b2, activation)
#         if pred == np.argmax(Y_test_data[i]):
#             bonnes += 1
#     taux = 100 * bonnes / len(I_test_data)
#     print(f"Précision avec {activation} : {taux:.2f}%")
#     return taux

# # Exemple d'appel :
# precision(I_test_data, Y_test_data, W1, bias1, W2, bias2, activation="sigmoide")



