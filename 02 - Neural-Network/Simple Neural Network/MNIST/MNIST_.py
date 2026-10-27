# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ------------ Paramètres ------------ 

epochs = 15
learning_rate = 0.01

#choix de la méthode utilisée selon stochastic, full_batch, mini_batch pour la descente

methode = "full_batch"
batch = 128 # généralement batch = 16, 32 ou 128

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

from tensorflow.keras.datasets import mnist
(I_train_data, Y_train_data), (I_test_data, Y_test_data) = mnist.load_data()  #I_train_data sont les matrices des nombres et Y_train_data les numéro


# ------------ Affichage de nombres pour l'exemple ------------ 

def affichage_MNIST(nb):
    """ 
        affiche l'image de l'item numéro nb de la banque d'image MNIST
    """
    plt.imshow(I_train_data[nb], cmap='gray')
    plt.colorbar()  #ajoute une échelle de couleurs
    plt.title('Image numéro : '+ str(nb))
    plt.show()


# ------------ Vectorisation ------------ 

#On transforme les matrices de MNIST en lignes
Data_set = []

for i in range(len(I_train_data)):
    N_image = []
    N_sortie = [0 for i in range(10)]
    for j in range(28):
        for k in range(28):
            N_image.append((I_train_data[i][j][k])/255) #on divise par 255 pour normalisée
            
    N_sortie[Y_train_data[i]] = 1
    
    Data_set.append((np.array(N_image), np.array(N_sortie)))
    
print("✅ Vectorisation effectuée")


# ------------ Implémentation ------------ 

def initialisation():
    """
        On initialise le model de variables (weights, bias)
    """
    W = np.random.randn(784, 10)
    bias = np.zeros(10)
    return W, bias

def pre_activation(I,W,bias):
    return I.dot(W)+bias

def activation(Z, fonction_activation):
    """
        Définition de la fonction sigmoïde
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
        
        exp_Z = np.exp(Z - np.max(Z))  # pour stabilité numérique et éviter l'overflow (astuce de calucul)
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
        return np.mean(- (1/len(S)) * np.sum(S*np.log(A+epsilon)+(1-S)*np.log(1-A+epsilon)))
    
    elif fonction_activation == "softmax":
        return -np.sum(S * np.log(A))  # produit de Hadamard

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
            cout = []
            
            # On mélange les données pour ne pas faire systématiquement la même descente
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
            
            # On mélange les données pour ne pas faire systématiquement la même descente
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

#Initialisation de l'algorithme : 

features = np.array([x[0] for x in Data_set])
tag = np.array([x[1] for x in Data_set])

W , bias = initialisation()

# Récupération des données après algorithme

W , bias , evo_cout = train(features, tag, W, bias, epochs, learning_rate, methode, batch, fonction_activation)

np.savetxt("weights_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + ".txt", W)
np.savetxt("bias_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + ".txt", bias)

print("✅ Poids et bias sauvegardés")


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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
