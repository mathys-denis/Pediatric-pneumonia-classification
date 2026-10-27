# -*- coding: utf-8 -*-

from PIL import Image
import glob
import os
import numpy as np

nbpix = 28

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

nompoids = input('Nom du fichier des poids (sans le txt) : ')
W = np.loadtxt(nompoids + '.txt')  # W.shape = (784, 2)
nombias = input('Nom du fichier des bias (sans le txt) : ')
bias = np.loadtxt(nombias + '.txt')

print("✅ Poids et bias chargés")

# ------------ Fonction de classification ------------

def calque(I):
    """Classification par produit scalaire + argmax"""
    C = I.dot(W) + bias
    # Pour softmax, argmax suffit
    return np.argmax(C)

# ------------ Calcul de la précision ------------

Nb_Erreur = 0
for I, Y in Data_set:
    if calque(I) != np.argmax(Y):
        Nb_Erreur += 1

precision = 100 * (len(Data_set) - Nb_Erreur) / len(Data_set)
print(f"Précision sur le jeu de test : {precision:.2f}%")
