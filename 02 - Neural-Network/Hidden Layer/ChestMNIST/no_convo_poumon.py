# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

# ------------ Paramètres ------------ 

epochs = 10
learning_rate = 0.05
methode = "mini_batch"  # "stochastic", "full_batch", "mini_batch"
batch = 32  # batch size
hidden_size = 64
fonction_activation = "sigmoide"  #idéal pour le multi-label

# ------------ Vérification des paramètres ------------ 

assert type(epochs) == int
assert type(learning_rate) in [float,int]
assert methode in ["stochastic","full_batch","mini_batch"]
assert type(batch) == int
assert fonction_activation in ["sigmoide","softmax"]
assert type(hidden_size) == int
print("✅ Aucun problème pour les entrées")

# ------------ Téléchargement des données ChestMNIST ------------ 

data_flag = "chestmnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

# ------------ Vectorisation ------------ 

Data_set = []
for i in range(len(train_dataset)):
    N_image = train_dataset.imgs[i].flatten().astype("float32") / 255.0
    N_sortie = train_dataset.labels[i].flatten()  # 14 labels multi-label
    Data_set.append((N_image, N_sortie))

print("✅ Vectorisation effectuée")
    
# ------------ Fonctions du modèle ------------ 

def initialisation(hidden_size, input_size=784, output_size=14):
    
    #Initialisation de la première couche
    W1 = np.random.randn(input_size, hidden_size)
    bias1 = np.zeros(hidden_size)
    
    #Initialisation de la deuxième couche
    W2 = np.random.randn(hidden_size, output_size)
    bias2 = np.zeros(output_size)
    
    return W1, bias1, W2, bias2

def pre_activation(I, W, bias):
    
    return I.dot(W) + bias

def activation(Z, fonction_activation):
    
    if fonction_activation == "sigmoide": 
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    elif fonction_activation == "softmax":
        exp_Z = np.exp(Z - np.max(Z))
        return exp_Z / np.sum(exp_Z)

def model(I, W, bias, fonction_activation):
    Z = pre_activation(I, W, bias)
    return activation(Z, fonction_activation)

def Loss(A, S, fonction_activation):
    epsilon = 1e-15
    if fonction_activation == "sigmoide":
        # binary cross-entropy multi-label
        return -np.mean(S*np.log(A+epsilon) + (1-S)*np.log(1-A+epsilon))
    elif fonction_activation == "softmax":
        return -np.mean(np.sum(S * np.log(A + epsilon)))

def gradient(W1, bias1, W2, bias2, I, S, fonction_activation):
    A1 = model(I, W1, bias1, "sigmoide")
    A2 = model(A1, W2, bias2, fonction_activation)

    dZ2 = A2 - S
    dW2 = np.outer(A1, dZ2)
    db2 = dZ2

    dZ1 = np.dot(W2, dZ2) * A1 * (1 - A1)
    dW1 = np.outer(I, dZ1)
    db1 = dZ1

    return dW1, db1, dW2, db2

def update(dW1, W1, db1, bias1, dW2, W2, db2, bias2, learning_rate):
    W1 = W1 - learning_rate * dW1
    bias1 = bias1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    bias2 = bias2 - learning_rate * db2
    return W1, bias1, W2, bias2

# ------------ Entrainement ------------ 

features = np.array([x[0] for x in Data_set])
tag = np.array([x[1] for x in Data_set])
W1, bias1, W2, bias2 = initialisation(hidden_size)

def train(features, tag, W1, bias1, W2, bias2, epochs, learning_rate, methode, batch, fonction_activation):
    n = len(features)
    evo_cout = []
    
    for epoch in range(epochs):
        print("Epoch ", epoch+1, "/", epochs, " starting -")
        cout = []
        indices = np.random.permutation(n)
        features_shuffled = features[indices]
        tag_shuffled = tag[indices]

        for i in range(0, n, batch):
            batch_features = features_shuffled[i:i+batch]
            batch_tags = tag_shuffled[i:i+batch]

            weights_gradient1 = np.zeros(W1.shape)
            bias_gradient1 = np.zeros(bias1.shape)
            weights_gradient2 = np.zeros(W2.shape)
            bias_gradient2 = np.zeros(bias2.shape)

            for feature, target in zip(batch_features, batch_tags):
                A1 = model(feature, W1, bias1, "sigmoide")
                A2 = model(A1, W2, bias2, fonction_activation)
                cout.append(Loss(A2, target, fonction_activation))
                dW1, db1, dW2, db2 = gradient(W1, bias1, W2, bias2, feature, target, fonction_activation)
                weights_gradient1 += dW1
                bias_gradient1 += db1
                weights_gradient2 += dW2
                bias_gradient2 += db2

            W1, bias1, W2, bias2 = update(weights_gradient1/len(batch_features),
                                          W1,
                                          bias_gradient1/len(batch_features),
                                          bias1,
                                          weights_gradient2/len(batch_features),
                                          W2,
                                          bias_gradient2/len(batch_features),
                                          bias2,
                                          learning_rate)
        evo_cout.append(np.mean(cout))
        print("  Erreur moyenne :", np.mean(cout))
    return W1, bias1, W2, bias2, evo_cout

W1, bias1, W2, bias2, evo_cout = train(features, tag, W1, bias1, W2, bias2,
                                       epochs, learning_rate, methode, batch, fonction_activation)

np.savetxt("med-non-convo-weights1_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", W1)
np.savetxt("med-non-convo-bias1_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", bias1)
np.savetxt("med-non-convo-weights2_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", W2)
np.savetxt("med-non-convo-bias2_" + str(epochs) + "_" + fonction_activation + "_" + str(learning_rate) + "_" + methode + "_" + str(batch) + "CC" + str(hidden_size) + ".txt", bias2)

print("✅ Poids et bias sauvegardés")

# ------------ Courbe de l'erreur ------------ 
plt.plot(range(len(evo_cout)), evo_cout, label="Erreur")
plt.xlabel("Epochs")
plt.ylabel("Erreur moyenne")
plt.title("Évolution de l'erreur (ChestMNIST)")
plt.legend()
plt.show()

