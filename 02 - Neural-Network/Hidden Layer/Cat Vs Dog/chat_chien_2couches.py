# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# ------------ Paramètres ------------ 

epochs = 20
learning_rate = 0.001

#choix de la méthode utilisée selon stochastic, full_batch, mini_batch pour la descente

methode = "mini_batch"
batch = 32 # généralement batch = 16, 32 ou 128

#choix de la fonction d'activation parmis sigmoide et softmax pour la dernière couche
#(la fonction d'activation pour la première couche est toujours sigmoïde)
fonction_activation = "softmax"

#choix du nombre de neuronnes dans la couche cachée
hidden_size = 128

#Redimensionnement des images :
nbpix = 32

# ------------ Vérification des paramètres ------------ 

assert type(epochs) == int, "epochs doit être un entier"
assert type(learning_rate) == float or type(learning_rate) == int, "le learning rate doit être un nombre"
assert methode in ["stochastic","full_batch","mini_batch"], "erreur sur la methode"
assert type(batch) == int, "batch doit être un entier"
assert fonction_activation in ["sigmoide","softmax"]
assert type(hidden_size) == int, "hidden_size doit être un entier"
assert type(nbpix) == int, "Nombre de pixels doit être un entier"


print("✅ Aucun problème pour les entrées")


# ------------ Téléchargement des données  ------------ 

train_dir = r"C:\Users\23mat\OneDrive\Bureau\TIPE\CONCOURS TIPE\data_set_dog_cat\archive\PetImages" #chemin du dossier
categories = ["Cat", "Dog"]

features = []
targets = []

for label, category in enumerate(categories):
    
    path = os.path.join(train_dir, category, "*.jpg")  # ouvre toutes les images .jpeg de la category
    
    for file in glob.glob(path):
        try:
            img = Image.open(file).resize((nbpix,nbpix))
            img = img.convert('L')  
            features.append(np.array(img).flatten())  # flatten indispensable (28x28 -> 784)
            if label == 0:
                targets.append(np.array([1,0])) #cas Cat
            else:
                targets.append(np.array([0,1])) #cas Dog
      
        except:
             print("""/!\\ Problème dans l'importation des image /!\\""")


features_ = np.array(features[:20000]) / 255.0 # Normalisation
print(len(features_))
targets_ = np.array(targets[:20000])

indices = np.random.permutation(len(features_)) # On mélange
features_ = features_[indices]
targets_ = targets_[indices]

Data_set = [(x,y) for x,y in zip (features_,targets_)]
print("✅ Téléchargement effectué")


# ------------ Implémentation ------------ 

def initialisation(hidden_size):
    """
        On initialise le model de variables (weights, bias)
    """

    #Initialisation de la première couche
    W1 = np.random.randn(nbpix**2, hidden_size)
    bias1 = np.zeros(hidden_size)
    
    #Initialisation de la seconde couche
    W2 = np.random.randn(hidden_size, 2)
    bias2 = np.zeros(2)
    
    return W1, bias1, W2, bias2

def pre_activation(I,W,bias):
    return I.dot(W)+bias

def activation(Z, fonction_activation):
    """
        Définition de la fonction sigmoïde / softmax
    """
    
    if fonction_activation == "sigmoide": 
        
        #----------- Sigmoïde -----------
        
        #La fonction np.clip permet de limiter les valeurs prisent dans l'exp 
        #ainsi on évite l'erreur d'overflow.
        #Par ailleurs, cela ne change rien car sigmoïde aurait renvoyée 0 ou 1
        
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    elif fonction_activation == "softmax":
        
        #----------- Softmax -----------
        
        exp_Z = np.exp(Z - np.max(Z))  # pour stabilité numérique et éviter l'overflow (astuce de calcul)
        return exp_Z / np.sum(exp_Z)

def model(I,W,bias,fonction_activation):
    Z = pre_activation(I, W, bias)
    return activation(Z,fonction_activation)

def Loss(A, S, fonction_activation):
    """ 
        Définition de la Log_Loss
    """
    epsilon = 1e-15  # petite constante pour éviter log(0)
    
    if fonction_activation == "sigmoide":
        
        # Cross-entropy binaire
        return -np.mean(S*np.log(A+epsilon)+(1-S)*np.log(1-A+epsilon))

    
    elif fonction_activation == "softmax":
        
        #Cross-entropy multi-classes
        return -np.mean(np.sum(S * np.log(A)))  # produit de Hadamard

def gradient(W1, bias1, W2, bias2, I, S, fonction_activation):
   
    A1 = model(I, W1, bias1, "sigmoide")
    A2 = model(A1, W2, bias2, fonction_activation)

    # Back-propagation 
    
    # Descente sur la seconde couche
    dZ2 = A2 - S
    dW2 = np.outer(A1,dZ2)
    db2 = dZ2
    
    # Descente sur la première couche
    dZ1 = np.dot(W2, dZ2) * A1 * (1 - A1)  # produit matriciel
    dW1 = np.outer(I, dZ1) # ecriture python de I^T.dZ1
    db1 = dZ1

    return dW1, db1, dW2, db2

def update(dW1, W1, db1, bias1, dW2, W2, db2, bias2, learning_rate):
    
    W1 = W1 - learning_rate * dW1
    bias1 = bias1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    bias2 = bias2 - learning_rate * db2
    
    return W1 , bias1, W2, bias2

def train(features,tag,W1,bias1,W2,bias2,epochs,learning_rate,methode,batch,fonction_activation):
    """ 
        descente du gradient
    """
    n = len(features)
    evo_cout = [] #évolution de l'erreur
    
    if methode == "full_batch":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            cout = []
            
            weights_gradient1 = np.zeros(W1.shape)
            bias_gradient1 = np.zeros(bias1.shape)
            weights_gradient2 = np.zeros(W2.shape)
            bias_gradient2 = np.zeros(bias2.shape)
            
            for feature, target in zip(features,tag):
                
                A1 = model(feature, W1, bias1, "sigmoide")
                A2 = model(A1, W2, bias2, fonction_activation)
                cout.append(Loss(A2,target,fonction_activation))
                
                dW1, db1, dW2, db2 = gradient(W1, bias1, W2, bias2, feature, target, fonction_activation)
                weights_gradient1 += dW1
                bias_gradient1 += db1
                weights_gradient2 += dW2
                bias_gradient2 += db2
                
            W1 , bias1, W2, bias2 = update(weights_gradient1/n, W1, bias_gradient1/n, bias1, weights_gradient2/n, W2, bias_gradient2/n, bias2, learning_rate)
            
            evo_cout.append(np.mean(cout))
            
        return W1, bias1, W2, bias2, evo_cout
    
    elif methode == "stochastic":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            cout = []
            
            # On mélange les données pour ne pas faire systématiquement la même descente
            indices = np.random.permutation(n)
            features = features[indices]
            tag = tag[indices]

            for feature, target in zip(features,tag):
                
                A1 = model(feature, W1, bias1, "sigmoide")
                A2 = model(A1, W2, bias2, fonction_activation)
                cout.append(Loss(A2,target,fonction_activation))
                
                dW1, db1, dW2, db2 = gradient(W1, bias1, W2, bias2, feature, target, fonction_activation)
                W1 , bias1, W2, bias2 = update(weights_gradient1/n, W1, bias_gradient1/n, bias1, weights_gradient2/n, W2, bias_gradient2/n, bias2, learning_rate)
                        
        return W1, bias1, W2, bias2, evo_cout
    
    elif methode == "mini_batch":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            cout = []
            
            # On mélange les données pour ne pas faire systématiquement la même descente
            indices = np.random.permutation(n)
            features = features[indices]
            tag = tag[indices]

            
            for i in range(0, n, batch):
                batch_features = features[i:i + batch]
                batch_tags = tag[i:i + batch]
            
                weights_gradient1 = np.zeros(W1.shape)
                bias_gradient1 = np.zeros(bias1.shape)
                weights_gradient2 = np.zeros(W2.shape)
                bias_gradient2 = np.zeros(bias2.shape)
            
                for feature, target in zip(batch_features,batch_tags):
                    
                    A1 = model(feature, W1, bias1, "sigmoide")
                    A2 = model(A1, W2, bias2, fonction_activation)
                    cout.append(Loss(A2,target,fonction_activation))
                    
                    dW1, db1, dW2, db2 = gradient(W1, bias1, W2, bias2, feature, target, fonction_activation)
                    weights_gradient1 += dW1
                    bias_gradient1 += db1
                    weights_gradient2 += dW2
                    bias_gradient2 += db2
                    
                W1 , bias1, W2, bias2 = update(weights_gradient1/len(batch_features), W1, bias_gradient1/len(batch_features), bias1, weights_gradient2/len(batch_features), W2, bias_gradient2/len(batch_features), bias2, learning_rate)
            
                evo_cout.append(np.mean(cout))
            
        return W1, bias1, W2, bias2, evo_cout

# ------------ Entrainement ------------ 

#Initialisation de l'algorithme : 

features = np.array([x[0] for x in Data_set])
tag = np.array([x[1] for x in Data_set])

W1, bias1, W2, bias2 = initialisation(hidden_size)

# Récupération des données après algorithme

W1, bias1, W2, bias2, evo_cout = train(features, tag, W1, bias1, W2, bias2, epochs, learning_rate, methode, batch, fonction_activation)

np.savetxt("weights1_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", W1)
np.savetxt("bias1_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", bias1)
np.savetxt("weights2_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", W2)
np.savetxt("bias2_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", bias2)

print("✅ Poids et bias sauvegardés")



# ------------ Interface graphique ------------ 

def augment_np(img_np):
    """
        Augmentation simple et compatible avec la vectorisation
        img_np : numpy array (nbpix*nbpix,) ou (nbpix,nbpix)
    """

    # Remise en forme si image flatten
    if img_np.ndim == 1:
        img_np = img_np.reshape(nbpix, nbpix)

    # Flip horizontal
    if np.random.rand() < 0.5:
        img_np = np.fliplr(img_np)

    # Variation de luminosité
    img_np = img_np + np.random.uniform(-0.1, 0.1)
    img_np = np.clip(img_np, 0.0, 1.0)

    # Bruit gaussien
    img_np = img_np + np.random.normal(0, 0.02, img_np.shape)
    img_np = np.clip(img_np, 0.0, 1.0)

    # Rotation légère (avec PIL)
    pil = Image.fromarray((img_np * 255).astype("uint8"))
    angle = np.random.uniform(-10, 10)
    pil = pil.rotate(angle, resample=Image.BILINEAR)

    # Toujours revenir à nbpix × nbpix
    pil = pil.resize((nbpix, nbpix))

    # Retour numpy normalisé entre 0 et 1 + flatten
    img_np = np.array(pil).astype("float32") / 255.0

    return img_np.flatten()


# ------------ Interface graphique ------------ 

def courbe_de_erreur(methode,fonction_activation, learning_rate, batch, hidden_size):
    
    if methode == "full_batch":
        
        info = "learning rate = " + str(learning_rate) + "\n" + \
       "méthode : " + methode + "\n" + \
       str(hidden_size) + " neuronnes dans la couche cachée" + "\n" + \
       "fonction d'activation : " + fonction_activation

        plt.plot(range(len(evo_cout)), evo_cout, label=info)
        plt.title("Evolution de l'erreur")
        plt.xlabel("n ième epoch")
        plt.legend()
        plt.show()
        
    elif methode == "mini_batch":
        
        info = "learning rate = " + str(learning_rate) + "\n" + \
            "méthode : " + methode + "\n" + \
            str(hidden_size) + " neuronnes dans la couche cachée" + "\n" + \
            "batch = " + str(batch) + "\n" + \
            "fonction d'activation : " + fonction_activation
        
        plt.plot(range(len(evo_cout)), evo_cout, label=info)
        plt.title("Evolution de l'erreur")
        plt.xlabel("n ième batch")
        plt.legend()
        plt.show()
        
        plt.figure()
        window_size = 100
        smooth_error = np.convolve(evo_cout, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(range(len(smooth_error)), smooth_error)
        plt.xlabel("Itération")
        plt.ylabel("Erreur moyenne")
        plt.title("Erreur lissée sur "+str(window_size)+" bachs")
        plt.show()
        
    elif methode == "stochastic":
        
        info = "learning rate = " + str(learning_rate) + "\n" + \
            "méthode : " + methode + "\n" + \
            str(hidden_size) + " neuronnes dans la couche cachée" + "\n" + \
            "fonction d'activation : " + fonction_activation
        
        plt.plot(range(len(evo_cout)), evo_cout, label=info)
        plt.title("Evolution de l'erreur")
        plt.xlabel("n ième itération")
        plt.legend()
        plt.show()
        
        plt.figure()
        window_size = 1000
        smooth_error = np.convolve(evo_cout, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(range(len(smooth_error)), smooth_error)
        plt.xlabel("Itération")
        plt.ylabel("Erreur moyenne")
        plt.title("Erreur lissée sur "+str(window_size)+" itérations")
        plt.show()
        
courbe_de_erreur(methode, fonction_activation, learning_rate, batch, hidden_size)
    
