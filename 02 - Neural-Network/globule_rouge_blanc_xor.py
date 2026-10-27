# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ------------ Implémentation ------------ 

  
def dataset():
    """
        Définissons notre dataset composé de deux classes:
            -Malade 1
            -Sain 0
    """

    #Nombre de personnes dans une classe
    row_per_class = 100
    
    #Generate rows
    sick = np.random.randn(row_per_class,2) + np.array([-2,-2]) #Les personnes malades ont des taux faibles
    healthy = np.random.randn(row_per_class,2) + np.array([2,2]) #Les personnes en bonne santé ont des taux élevés
    
    features = np.vstack([sick, healthy]) #On met la matrice de l'une en dessous de l'autre, shape (10,2)
    tag = np.concatenate((np.ones(row_per_class), np.zeros(row_per_class))) #On étiquettes nos données/features
    
    return features, tag

def dataset_xor():
    """
        Dataset non linéairement séparable :
            - Sain (0) : globules rouges hauts / blancs bas OU rouges bas / blancs hauts
            - Malade (1) : globules rouges et blancs tous deux hauts OU tous deux bas
    """

    row_per_quadrant = 50

    # Groupe 1 : Malade (taux faibles)
    sick_low = np.random.randn(row_per_quadrant, 2) + np.array([-2, -2])

    # Groupe 2 : Malade (taux élevés)
    sick_high = np.random.randn(row_per_quadrant, 2) + np.array([2, 2])

    # Groupe 3 : Sain (rouges hauts / blancs bas)
    healthy_rb = np.random.randn(row_per_quadrant, 2) + np.array([2, -2])

    # Groupe 4 : Sain (rouges bas / blancs hauts)
    healthy_br = np.random.randn(row_per_quadrant, 2) + np.array([-2, 2])

    # Fusion des données
    features = np.vstack([sick_low, sick_high, healthy_rb, healthy_br])
    tag = np.concatenate((
        np.ones(2 * row_per_quadrant),      # Malades = 1
        np.zeros(2 * row_per_quadrant)      # Sains = 0
    ))

    return features, tag

def affichage_globule_blanc_rouge (features,tag):
    """
         schéma du dataset
    """
    
    for i, elt in enumerate(features):
        if tag[i] ==1:
            plt.plot(elt[0],elt[1],'rx')
        else:
            plt.plot(elt[0],elt[1],'gx')
    plt.xlabel('Quantité de Globules rouges')
    plt.ylabel('Quantité de Globules blancs')
    plt.legend()
    
def afficher_frontiere(features, weights1, bias1, weights2, bias2, resolution=200):
    # Grille de points
    
    plt.figure()
    x_min, x_max = features[:,0].min() - 1, features[:,0].max() + 1
    y_min, y_max = features[:,1].min() - 1, features[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Passage dans le réseau
    Z1 = np.dot(grid, weights1) + bias1
    A1 = activation(Z1)
    Z2 = np.dot(A1, weights2) + bias2
    A2 = activation(Z2).reshape(xx.shape)

    # Tracé
    plt.contourf(xx, yy, A2, levels=50, cmap="RdYlGn", alpha=0.6)
    plt.colorbar(label="Probabilité d'être malade")
    affichage_globule_blanc_rouge(features, tag)
    plt.title("Frontière de décision (réseau à 2 couches)")
    plt.show()
    
def init_variables():
    """
        On initialise le model de variables (weights, bias)
    """
    
    # 1ère couche
    weights1 = np.random.normal(size=(2,3))
    bias1 = np.zeros(3)
    
    # 2ème couche 
    weights2 = np.random.normal(size=(3,1))
    bias2 = np.zeros(1)

    return weights1, bias1, weights2, bias2

def pre_activation(features,weights,bias):
    return np.dot(features,weights)+bias

def activation(z):
    """
        Définition de la fonction sigmoïde
    """
    
    return 1/(1+np.exp(-z))
    
def predict(features,w1,b1,w2,b2):
    """ 
        Calcul de la probabilité d'appartenir à une classe ou une autre
    """
    A1 = activation(np.dot(features, w1) + b1)
    A2 = activation(np.dot(A1, w2) + b2)
    return (A2 > 0.5).astype(int)

        
def erreur(predictions, tag):
    predictions = predictions.reshape(-1) #on s'assure de la bonne dimension
    epsilon = 1e-15  # petite constante pour éviter log(0)
    LL = - (1/len(tag)) * np.sum(tag*np.log(predictions+epsilon)+(1-tag)*np.log(1-predictions+epsilon))
    return LL

def train(features,tag,weights1,bias1,weights2,bias2,epochs):
    """ 
        Descente du gradient
    """
    learning_rate = 0.1
    pred = [] #évolution du diagnostique pour chaque patient
    cout = [] #évolution de l'erreur
    just_diag = [] #justesse du diagnostique en %
    w2 = [] #poids sauvegardés de la couche 2
    b2 = [] #bias sauvegardés de la couche 2
    w1 = [] #poids sauvegardés de la couche 1
    b1 = [] #bias sauvegardés de la couche 1
    
    for epoch in range(epochs):
        
        # Récupération de données pour les graphs
        
        w1.append(weights1)
        b1.append(bias1)
        w2.append(weights2)
        b2.append(bias2)
        
        Z1 = np.dot(features, weights1) + bias1
        A1 = activation(Z1)
        Z2 = np.dot(A1, weights2) + bias2
        A2 = activation(Z2)
        
        predictions = (A2 > 0.5).astype(int)
        just_diag.append(np.mean(predictions.reshape(-1) == tag.reshape(-1))) #on s'assure des bonnes dimensions)
        cout.append(erreur(A2,tag))
        pred.append(A2.mean())

            
        # Init. gradients et bias

        weights_gradient1 = np.zeros(weights1.shape)
        weights_gradient2 = np.zeros(weights2.shape)
        bias_gradient1 = np.zeros(bias1.shape)
        bias_gradient2 = np.zeros(bias2.shape)
        
        # Descente 
        
        for feature, target in zip(features,tag):
            
            Z1 = np.dot(feature, weights1) + bias1 #pre_activation
            A1 = activation(Z1)
            Z2 = np.dot(A1, weights2) + bias2 #pre_activation
            A2 = activation(Z2)
            
            # Descente sur la seconde couche
            dZ2 = A2 - target
            weights_gradient2 += np.outer(A1,dZ2)
            bias_gradient2 += dZ2
            
            # Descente sur la première couche
            dZ1 = np.dot(weights2, dZ2) * A1 * (1 - A1)  # produit matriciel
            weights_gradient1 += np.outer(feature, dZ1)
            bias_gradient1 += dZ1
            
        # Normalisation 
        
        weights_gradient2 = weights_gradient2 * (1/len(tag))
        bias_gradient2 = bias_gradient2 * (1/len(tag))
        weights_gradient1 = weights_gradient1 * (1/len(tag))
        bias_gradient1 = bias_gradient1 * (1/len(tag))
        
        weights2 = weights2 - learning_rate*weights_gradient2
        bias2 = bias2 - learning_rate*bias_gradient2
        weights1 = weights1 - learning_rate*weights_gradient1
        bias1 = bias1 - learning_rate*bias_gradient1
        
    
    w1.append(weights1)
    b1.append(bias1)
    w2.append(weights2)
    b2.append(bias2)
    
    return cout, pred, just_diag, w1, b1, w2, b2
            

# ------------ Entrainement ------------ 

epochs = 10000

# Initialistation de l'algorithme

features, tag = dataset_xor()
weights1_i, bias1_i, weights2_i, bias2_i = init_variables()

# Récupération des données après algorithme

cout, pred, just_diag, w1, b1, w2, b2 = train(features, tag, weights1_i, bias1_i, weights2_i, bias2_i, epochs)


# ------------ Interface graphique ------------ 


affichage_globule_blanc_rouge(features, tag)
plt.xlim(-5,5)
plt.ylim(-6,6)
plt.show()

afficher_frontiere(features, weights1_i, bias1_i, weights2_i, bias2_i)

afficher_frontiere(features, w1[epochs//2], b1[epochs//2], w2[epochs//2], b2[epochs//2])

afficher_frontiere(features, w1[-1], b1[-1], w2[-1], b2[-1])

epoc = [i + 1 for i in range(epochs)]
plt.plot(epoc, cout, '-')
plt.title("Évolution de l'erreur")
plt.xlabel("Époch")
plt.ylabel("Erreur (log loss)")
plt.show()


plt.figure()
plt.plot(just_diag)
plt.title("Évolution de la justesse du diagnostic")
plt.xlabel("Époch")
plt.ylabel("Précision (%)")
plt.show()


# ------------ Test des poids obtenues ------------ 

nb = 1000  # nombre de tests
precisions = []

for i in range(nb):
    features_test, tag_test = dataset_xor()
    predictions = predict(features_test, w1[-1], b1[-1], w2[-1], b2[-1])  # utiliser les poids finaux
    precision = np.mean(predictions.reshape(-1) == tag.reshape(-1)) #on s'assure des bonnes dimensions
    precisions.append(precision)

print("L'algorithme est fiable à " + str(np.mean(precisions)*100) + "%")














