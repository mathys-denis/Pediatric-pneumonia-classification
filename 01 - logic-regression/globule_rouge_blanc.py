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
    plt.figure()
    
    for i, elt in enumerate(features):
        if tag[i] ==1:
            plt.plot(elt[0],elt[1],'rx')
        else:
            plt.plot(elt[0],elt[1],'gx')
    plt.xlabel('Quantité de Globules rouges')
    plt.ylabel('Quantité de Globules blancs')
    plt.legend()
    
def visualiser_descente(features, tag, w_hist, b_hist):
    """
    Visualise la surface de la fonction de coût et la descente du gradient.
    """
    # Trajectoire des poids
    traj_w1 = [w[0] for w in w_hist]
    traj_w2 = [w[1] for w in w_hist]

    # Plage de valeurs autour de la trajectoire
    pad = 0.5
    w1_min, w1_max = min(traj_w1)-pad, max(traj_w1)+pad
    w2_min, w2_max = min(traj_w2)-pad, max(traj_w2)+pad
    w1_vals = np.linspace(w1_min, w1_max, 50)
    w2_vals = np.linspace(w2_min, w2_max, 50)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    # On fixe le biais au biais initial (coupe 3D)
    b_fix = b_hist[0]
    Z = np.zeros_like(W1)

    # Calcul du coût pour chaque (w1, w2)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            weights_tmp = np.array([W1[i, j], W2[i, j]])
            Z[i, j] = erreur(activation(pre_activation(features, weights_tmp, b_fix)), tag)

    # Calcul du coût réel à chaque étape (pour la trajectoire)
    traj_z = [
        erreur(activation(pre_activation(features, np.array([w[0], w[1]]), b)), tag)
        for w, b in zip(w_hist, b_hist)
    ]

    # --- Affichage ---
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)

    # Trajectoire
    ax.plot(traj_w1, traj_w2, traj_z, 'r-', linewidth=3, label='Descente réelle')
    ax.scatter(traj_w1[0], traj_w2[0], traj_z[0], c='g', s=80, label='Départ (init)')
    ax.scatter(traj_w1[-1], traj_w2[-1], traj_z[-1], c='r', s=80, label='Arrivée')

    # Réglages
    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_zlabel('Erreur')
    ax.set_zlim(0, 1)
    ax.set_title("Descente du gradient sur la surface de coût")
    ax.legend()
    plt.show()

    
def affichage_fontière(features, weights, bias):
    
    if weights is not None and bias is not None:
        x_vals = np.linspace(features[:, 0].min() - 1, features[:, 0].max() + 1, 200)
        y_vals = -(weights[0] / weights[1]) * x_vals - bias / weights[1]
        plt.plot(x_vals, y_vals, 'k-', label='Frontière de décision')
        plt.legend()
    
def init_variables():
    """
        On initialise le model de variables (weights, bias)
    """
    
    weights = np.random.normal(size=2)
    bias = 0
    return weights, bias

def pre_activation(features,weights,bias):
    return np.dot(features,weights)+bias

def activation(z):
    """
        Définition de la fonction sigmoïde
    """
    
    return 1/(1+np.exp(-z))
    
def predict(features,weights, bias):
    """ 
        Calcul de la probabilité d'appartenir à une classe ou une autre
    """
    z = pre_activation(features, weights, bias)
    a = activation(z)
    return(np.round(a))
        
def erreur(predictions, tag):
    epsilon = 1e-15  # petite constante pour éviter log(0)
    LL = - (1/len(tag)) * np.sum(tag*np.log(predictions+epsilon)+(1-tag)*np.log(1-predictions+epsilon))
    return LL

def train(features,tag,weights,bias):
    """ 
        Descente du gradient
    """
    epochs = 100
    learning_rate = 0.1
    pred = [] #évolution du diagnostique pour chaque patient
    cout = [] #évolution de l'erreur
    just_diag = [] #justesse du diagnostique en %
    w = [] #poids sauvegardés
    b = [] #bias sauvegardés
    
    for epoch in range(epochs):
        
        w.append(weights)
        b.append(bias)
            
        predictions = predict(features, weights, bias)
        just_diag.append(np.mean(predictions==tag))
        pred.append(activation(pre_activation(features, weights, bias)))
        cout.append(erreur(activation(pre_activation(features, weights, bias)),tag))
            
        # Init gradients
        
        weights_gradient = np.zeros(weights.shape)
        bias_gradient = 0
        
        for feature, target in zip(features,tag):
            z = pre_activation(feature, weights, bias)
            a = activation(z)
            weights_gradient += (a - target) * feature
            bias_gradient += (a - target)
            
        weights_gradient = weights_gradient * (1/len(tag))
        bias_gradient = bias_gradient * (1/len(tag))
        
        weights = weights - learning_rate*weights_gradient
        bias = bias - learning_rate*bias_gradient
    
    
    return cout, pred, just_diag, w, b
            

# ------------ Entrainement ------------ 


# Initialistation de l'algorithme

features, tag = dataset_xor()
weights, bias = init_variables()

# Récupération des données après algorithme

cout, pred, just_diag, w, b = train(features, tag, weights, bias)


# ------------ Interface graphique ------------ 


affichage_globule_blanc_rouge(features, tag)
plt.xlim(-5,5)
plt.ylim(-6,6)

affichage_globule_blanc_rouge(features, tag)
affichage_fontière(features, weights, bias)
plt.title("Situation à l'initialisation")
plt.xlim(-5,5)
plt.ylim(-6,6)
plt.legend()

affichage_globule_blanc_rouge(features, tag)
affichage_fontière(features, w[49], b[49])
plt.title("Situation à epoch=50")
plt.xlim(-5,5)
plt.ylim(-6,6)
plt.legend()

affichage_globule_blanc_rouge(features, tag)
affichage_fontière(features, w[99], b[99])
plt.title("Situation finale")
plt.xlim(-5,5)
plt.ylim(-6,6)
plt.legend()

epoc = [i + 1 for i in range(100)]
plt.figure()
plt.plot(epoc,cout,'-')
plt.plot(0,cout[0],'rx', label='erreur epoch 1 = ' + str(cout[0]))
plt.plot(9,cout[9],'gx', label='erreur epoch 10 = ' + str(cout[9]))
plt.plot(49,cout[49],'yx', label='erreur epoch 50 = ' + str(cout[49]))
plt.plot(99,cout[99],'mx', label='erreur epoch 100 = ' + str(cout[99]))
plt.title("évolution de l'erreur")
plt.xlabel('epoch')
plt.ylabel('erreur')
plt.legend()
plt.show()

plt.figure()
plt.plot(epoc,pred,'-')
plt.title("évolution du diagnostique pour chaque patient")
plt.xlabel('epoch')
plt.ylabel('%chance malade/sain')
plt.legend()
plt.show()

plt.figure()
plt.plot(epoc,just_diag,'-')
plt.plot(0,just_diag[0],'rx', label='justesse epoch 1 = ' + str(just_diag[0]))
plt.plot(9,just_diag[9],'gx', label='justesse epoch 10 = ' + str(just_diag[9]))
plt.plot(49,just_diag[49],'yx', label='justesse epoch 50 = ' + str(just_diag[49]))
plt.plot(99,just_diag[99],'mx', label='justesse epoch 100 = ' + str(just_diag[99]))
plt.title("évolution de la justesse du diagnostique")
plt.xlabel('epoch')
plt.ylabel('précision (%)')
plt.legend()
plt.show()

visualiser_descente(features, tag, w, b)

# ------------ Test des poids obtenues ------------ 

nb = 100  # nombre de tests
precisions = []

for i in range(nb):
    features_test, tag_test = dataset_xor()
    predictions = predict(features_test, w[-1], b[-1])  # utiliser les poids finaux
    precision = np.mean(predictions == tag_test)
    precisions.append(precision)

print("L'algorithme est fiable à " + str(np.mean(precisions)*100) + "%")













