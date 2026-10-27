# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

# ------------ Téléchargement des données ChestMNIST ------------
data_flag = "chestmnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

# Flatten et normalisation
Data_set = []
for i in range(len(test_dataset)):
    N_image = test_dataset.imgs[i].flatten().astype("float32") / 255.0
    Data_set.append(np.array(N_image))

print("✅ Données flattenées et normalisées prêtes")

# ------------ Chargement des poids et biais ------------
nompoids1 = input('Nom du fichier des poids 1 (sans le txt) : ')
W1 = np.loadtxt(nompoids1 + '.txt')

nombias1 = input('Nom du fichier des bias 1 (sans le txt) : ')
bias1 = np.loadtxt(nombias1 + '.txt')

nompoids2 = input('Nom du fichier des poids 2 (sans le txt) : ')
W2 = np.loadtxt(nompoids2 + '.txt')

nombias2 = input('Nom du fichier des bias 2 (sans le txt) : ')
bias2 = np.loadtxt(nombias2 + '.txt')

# ------------ Visualisation des filtres ------------

def afficher_filtre_didactique(filtre, shape=(28,28), titre=None):
    filtre_2D = filtre.reshape(shape)
    vabs = np.max(np.abs(filtre_2D))
    plt.imshow(filtre_2D, cmap='bwr', vmin=-vabs, vmax=vabs)
    plt.axis("off")
    if titre:
        plt.title(titre, fontsize=10, fontweight="bold")

def visualiser_filtres_didactique(W, shape=(28,28), n_affiche=10):
    plt.figure(figsize=(12, 6))
    for i in range(n_affiche):
        plt.subplot(2, n_affiche//2, i + 1)
        afficher_filtre_didactique(W[:, i], shape, titre=f"Filtre du neurone {i}")
    plt.suptitle(
        "Filtres appris — zones rouges = pixels activants, bleues = inhibants",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

visualiser_filtres_didactique(W1, n_affiche=min(W1.shape[1], 10))

def visualiser_W2(W2):
    plt.figure(figsize=(10, 4))
    plt.imshow(W2, cmap="bwr", aspect="auto")
    plt.colorbar()
    plt.title(f"Poids entre la couche cachée ({W2.shape[0]} neurones) et les {W2.shape[1]} sorties")
    plt.xlabel("Classe de sortie")
    plt.ylabel("Neurones cachés")
    plt.show()

visualiser_W2(W2)

# ------------ Fonction de prédiction ------------

def calque(I, W1, b1, W2, b2):
    Z1 = I.dot(W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))  # sigmoïde couche cachée
    Z2 = A1.dot(W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoïde multi-label
    return A2  # vecteur de probabilités pour chaque pathologie

# ------------ Prétraitement image individuelle ------------

def redim(img, f):
    return img.resize((f,f))

def convertgris(img):
    return np.array(img.convert("L")).astype(int)

def ajustpixels(array):
    return (255 - array)/255

def bonformat(img):
    img = redim(img, 28)
    img = convertgris(img)
    return img

def test_image():
    nom2 = input("Nom de l'image (sans extension) : ")
    ext = input("Format de l'image (ex: png, jpg) : ")
    try:
        inter = Image.open(f"{nom2}.{ext}")
        inter = ajustpixels(bonformat(inter))
        img = inter.flatten()
        pred = calque(img, W1, bias1, W2, bias2)
        print("Probabilités prédites :", pred)
        print("Labels prédits (seuil 0.5) :", (pred>0.5).astype(int))
    except:
        print("/!\\ Erreur : nom ou format de l'image incorrect")

# ------------ Test d’efficacité sur le dataset ------------

Nb_Erreur = 0
total_labels = len(Data_set) * 14

for i in range(len(Data_set)):
    pred = (calque(Data_set[i], W1, bias1, W2, bias2) > 0.5).astype(int)
    Nb_Erreur += np.sum(pred != test_dataset.labels[i].flatten())

precision = 100 * (total_labels - Nb_Erreur)/total_labels
print("Précision globale :", precision, "%")
