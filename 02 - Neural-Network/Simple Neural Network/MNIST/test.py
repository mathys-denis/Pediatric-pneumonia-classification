# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ------------ Téléchargement des données ------------

(I_train_data, Y_train_data), (I_test_data, Y_test_data) = mnist.load_data()

Data_set = []

for i in range(len(I_test_data)):
    N_image = []
    for j in range(28):
        for k in range(28):
            N_image.append((I_test_data[i][j][k])/255)
            
    Data_set.append(np.array(N_image))

# ------------ Chargement des poids et bias ------------

nompoids = input('Nom du fichier des poids (sans le txt) : ')
W = np.loadtxt(nompoids + '.txt')  # W.shape = (784, 10)

nombias = input('Nom du fichier des bias (sans le txt) : ')
bias = np.loadtxt(nombias + '.txt')

# ------------ Fonction pour afficher un filtre ------------

def afficher_filtre(poids, taille=(28, 28), titre=""):
    poids = np.array(poids).reshape(taille)
    max_abs = np.max(np.abs(poids))
    if max_abs == 0:
        max_abs = 1
    poids_norm = poids / max_abs

    img = np.zeros((*taille, 3))
    img[:, :, 0] = np.clip(poids_norm, 0, 1)   # Rouge
    img[:, :, 2] = np.clip(-poids_norm, 0, 1)  # Bleu
    img[np.isclose(poids_norm, 0)] = [1, 1, 1]  # Blanc

    plt.imshow(img)
    plt.title(titre)
    plt.axis('off')

# ------------ Affichage de tous les filtres ------------

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    afficher_filtre(W[:, i], (28, 28), titre=f"Classe {i}")
plt.suptitle("Visualisation des filtres appris", fontsize=14)
plt.tight_layout()
plt.show()


def afficher_filtre_didactique(filtre, shape=(28,28), titre=None):
    """
    Affiche un filtre (vecteur de poids) sous forme d'image :
    - rouge vif : active la classe
    - bleu foncé : inhibe la classe
    - blanc : neutre
    """
    filtre_2D = filtre.reshape(shape)
    vabs = np.max(np.abs(filtre_2D))
    
    plt.imshow(filtre_2D, cmap='bwr', vmin=-vabs, vmax=vabs)
    plt.axis("off")
    if titre:
        plt.title(titre, fontsize=10, fontweight="bold")

def visualiser_filtres_didactique(W, shape=(28,28)):
    """
    Affiche les 10 filtres appris (un par chiffre)
    """
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        afficher_filtre_didactique(W[:, i], shape, titre=f"Classe {i}")
    plt.suptitle(
        "Filtres appris — zones rouges = pixels activants, bleues = inhibants",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


visualiser_filtres_didactique(W)


# ------------ test image en particulier ------------

def calque(I):
    
    C = I.dot(W)+bias
    
    nb = 0
    for j in range (len(C)):
        if C[j]>C[nb]:
            nb = j
    
    return nb

def bonformat (img):
    """
        Entrée : Image
        
        Objectif : Mettre l'image dans un format adaptée aux tests
        
        Réalise : -Redimensionne l'image
                  -La convertit en un tableau avec des entiers compris entre 0 et 255
                  
        Sortie : Tableau
    """
    
    img = redim(img, 28)
    img = convertgris(img)
    
    return (img)

def redim(img, f):
    """
        Entrée : img = Image et f = entier positif 
        
        Objectif : Redimensionner une image dans le format "fxf"
        
        Sortie : Image
    """
    
    img_resize = img.resize((f, f))
    return (img_resize)

def convertgris (img):
    """
        Entrée : Image
        
        Objectif : Permet d'obtenir un tableau contenant le niveau de gris des pixels.
        
        Info : Le tableau renvoie ligne par ligne
        
        Sortie : Tableau
    """
    
    # Convertit en tableau NumPy (avec des nombres compris entre 0 et 255)
    #(0 correspond à noir, 255 à blanc):
    image = img.convert("L")
    
    
    #cette fonction sert à pouvoir ajuster les entiers dans la matrice 
    #et pas uniquement avoir des entiers entre 0 et 255
    array = np.array(image).astype(int)

    return (array)

def ajustpixels(array):
    """
        Entrée : Tableau
        
        Objectif : Changer les valeurs des pixels
        
        Realise : -Les nombres proches de 255 correspondant au blanc deviennent proche de 0
                  -Les nombres proches de 0 correspondant au noir deviennent proche de 1
                  
        Sortie : Tableau
    """
    
    return (255 - array)/255

def test():
    nom2 = input("nom de l'image:") + "." + input("format de l'image:")
    try:
        inter = Image.open(nom2)
        inter = list(ajustpixels(bonformat(inter)))
        img = []
        for l in range(28):
            for c in range(28):
                img.append(inter[l][c])
        print(calque(np.array(img)))
    
    except:
        print("/!\ Erreur : Le nom ou le format de l'image est incorrect")


# ------------ Efficacité ------------

Nb_Erreur = 0

for i in range(len(I_test_data)):
    
    if calque(Data_set[i]) != Y_test_data[i]:
        Nb_Erreur+=1

print("L'algorithme est précis à la hauteur de :", 100*(len(I_test_data)-Nb_Erreur)/len(I_test_data),"%")

