# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

nbpix = 150

# ------------ Téléchargement des données ------------

import os
import glob

train_dir = r"C:\Users\23mat\.kaggle\chest_xray_data\archive\chest_xray\test"
categories = ["NORMAL", "PNEUMONIA"]

I_test_data = []
Y_test_data = []

for label, category in enumerate(categories):

    path = os.path.join(train_dir, category, "*.jpeg")

    for file in glob.glob(path):
        try:
            img = Image.open(file).convert('L')
            img = img.resize((nbpix, nbpix))

            # ✅ FIX IMPORTANT : on vectorise directement ici
            img = np.array(img).flatten() / 255.0

            I_test_data.append(img)

            if label == 0:
                Y_test_data.append(np.array([1,0]))
            else:
                Y_test_data.append(np.array([0,1]))

        except:
            print("/!\\ Problème dans l'importation des images /!\\")

print("✅ Importation réussie")

I_test_data = np.array(I_test_data)
Y_test_data = np.array(Y_test_data)

# ------------ Chargement des poids et bias ------------

nompoids1 = input('Nom du fichier des poids 1 (sans le txt) : ')
W1 = np.loadtxt(nompoids1 + '.txt')

nombias1 = input('Nom du fichier des bias 1 (sans le txt) : ')
bias1 = np.loadtxt(nombias1 + '.txt')

nompoids2 = input('Nom du fichier des poids 2 (sans le txt) : ')
W2 = np.loadtxt(nompoids2 + '.txt')

nombias2 = input('Nom du fichier des bias 2 (sans le txt) : ')
bias2 = np.loadtxt(nombias2 + '.txt')


# ------------ test image ------------

def calque(I, W1, b1, W2, b2):
    """
    Réseau cohérent avec l'entraînement :
    - sigmoïde couche cachée
    - softmax sortie
    """

    # sécurité (normalement inutile maintenant)
    if I.ndim == 2:
        I = I.flatten()

    # --- couche 1 ---
    Z1 = I.dot(W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    # --- couche 2 ---
    Z2 = A1.dot(W2) + b2

    # softmax stable (IMPORTANT)
    expZ = np.exp(Z2 - np.max(Z2))
    A2 = expZ / np.sum(expZ)

    return np.argmax(A2)


# ------------ Efficacité ------------

Nb_Erreur = 0

for i in range(len(I_test_data)):

    pred = calque(I_test_data[i], W1, bias1, W2, bias2)

    if pred != np.argmax(Y_test_data[i]):
        Nb_Erreur += 1

print("L'algorithme est précis à la hauteur de :",
      100*(len(I_test_data)-Nb_Erreur)/len(I_test_data), "%")