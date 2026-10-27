# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# ------------ Paramètres ------------ 

nbpix = 28

epochs = 10
learning_rate = 0.05

#choix de la méthode utilisée selon stochastic, full_batch, mini_batch pour la descente

methode = "mini_batch"
batch = 16 # généralement batch = 16, 32 ou 128

#choix de la fonction d'activation parmis sigmoide et softmax
fonction_activation = "softmax"

# ------------ Vérification des paramètres ------------ 

assert type(epochs) == int, "epochs doit être un entier"
assert type(learning_rate) == float or type(learning_rate) == int, "le learning rate doit être un nombre"
assert methode in ["stochastic","full_batch","mini_batch"], "erreur sur la methode"
assert type(batch) == int, "batch doit être un entier"
assert fonction_activation in ["sigmoide","softmax"]

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

def initialisation():
    """
        On initialise le model de variables (weights, bias)
    """
    W = np.random.randn(784, 2)
    bias = np.zeros(2)
    return W, bias

def pre_activation(I,W,bias):
    return I.dot(W)+bias

def activation(Z, fonction_activation):
    """
        Définition de la fonction sigmoïde ou softmax
    """
    
    if fonction_activation == "sigmoide": 
        
        #----------- Sigmoïde -----------
        
        #La fonction np.clip permet de limiter les valeurs prisent dans l'exp 
        #ainsi on évite l'erreur d'overflow.
        #Par ailleurs, cela ne change rien car sigmoïde aurait renvoyé 0 ou 1
        
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    elif fonction_activation == "softmax":
        
        #----------- Softmax -----------
        
        exp_Z = np.exp(Z - np.max(Z))  # stabilité numérique
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
        
        # Cross-entropy binaire (forme correcte)
        return -np.sum(S*np.log(A+epsilon)+(1-S)*np.log(1-A+epsilon))
    
    elif fonction_activation == "softmax":
        return -np.sum(S * np.log(A + epsilon))  # cross-entropy

def gradient(W, I, S, bias, fonction_activation):
   
    A = model(I, W, bias, fonction_activation)

    dZ = A - S  
    dW = np.outer(I, dZ)  # ecriture python de I^T.dZ
    db = dZ
    return dW, db

def update(dW, W, bias, db, learning_rate):
    W = W - learning_rate * dW
    bias = bias - learning_rate * db
    return W , bias

def train(features,tag,W,bias,epochs,learning_rate,methode,batch,fonction_activation):
    """ 
        descente du gradient
    """
    n = len(features)
    evo_cout = [] #évolution de l'erreur
    
    if methode == "full_batch":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            cout = []
            
            weights_gradient = np.zeros(W.shape)
            bias_gradient = np.zeros(bias.shape)
            
            for feature, target in zip(features,tag):
                A = model(feature, W, bias,fonction_activation)
                cout.append(Loss(A,target,fonction_activation))
                dW , db = gradient(W,feature,target,bias,fonction_activation)
                weights_gradient += dW
                bias_gradient += db
                
            W , bias = update(weights_gradient/n, W, bias, bias_gradient/n, learning_rate)
            evo_cout.append(np.mean(cout))
            
        return W , bias , evo_cout
    
    elif methode == "stochastic":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            
            indices = np.random.permutation(n)
            features = features[indices]
            tag = tag[indices]

            for feature, target in zip(features,tag):
                A = model(feature, W, bias,fonction_activation)
                evo_cout.append(Loss(A,target,fonction_activation))
                dW , db = gradient(W,feature,target,bias,fonction_activation)
                W , bias = update(dW, W, bias, db, learning_rate)
                        
        return W , bias , evo_cout
    
    elif methode == "mini_batch":
        
        for epoch in range(epochs):
            
            print("Epoch ", epoch, "/", epochs, " starting -" )
            cout = []
            
            indices = np.random.permutation(n)
            features = features[indices]
            tag = tag[indices]

            for i in range(0, n, batch):
                batch_features = features[i:i + batch]
                batch_tags = tag[i:i + batch]
            
                weights_gradient = np.zeros(W.shape)
                bias_gradient = np.zeros(bias.shape)
            
                for feature, target in zip(batch_features,batch_tags):
                    A = model(feature, W, bias,fonction_activation)
                    cout.append(Loss(A,target,fonction_activation))
                    dW , db = gradient(W,feature,target,bias,fonction_activation)
                    weights_gradient += dW
                    bias_gradient += db
                
                W , bias = update(weights_gradient / len(batch_features), W, bias, bias_gradient / len(batch_tags), learning_rate)
                evo_cout.append(np.mean(cout))
            
        return W , bias , evo_cout

# ------------ Entrainement ------------ 

features = np.array([x[0] for x in Data_set])
tag = np.array([x[1] for x in Data_set])

W , bias = initialisation()

W , bias , evo_cout = train(features, tag, W, bias, epochs, learning_rate, methode, batch, fonction_activation)

np.savetxt("weights_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + ".txt", W)
np.savetxt("bias_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + ".txt", bias)

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

def courbe_de_erreur(methode,fonction_activation, learning_rate, batch):
    
    if methode == "full_batch":
        
        info = "learning rate = " + str(learning_rate) + "\n" + \
       "méthode : " + methode + "\n" + \
       "fonction d'activation : " + fonction_activation

        plt.plot(range(len(evo_cout)), evo_cout, label=info)
        plt.title("Evolution de l'erreur")
        plt.xlabel("n ième epoch")
        plt.legend()
        plt.show()
        
    elif methode == "mini_batch":
        
        info = "learning rate = " + str(learning_rate) + "\n" + \
            "méthode : " + methode + "\n" + \
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
        
courbe_de_erreur(methode, fonction_activation, learning_rate, batch)

