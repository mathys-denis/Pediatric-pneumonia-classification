import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy.ndimage import rotate, zoom


# ------------ Paramètres ------------ 

nbpix = 224 # dimension d'un côté d'une image
nbcol = 1 # Correspond au système de couleur (1 pour gris et 3 pour RGB)
nb_neuronne_couche_int = 128 #256
batch_size = 16
epochs = 45


# ------------ Vérification des paramètres ------------ 

assert type(nbpix) == int, "La dimension du côté d'une image doit être un entier"
assert type(nbcol) == int, "nbcol doit être un entier"
assert type(nb_neuronne_couche_int) == int, "Il doit y avoir un nombre entier de neurones sur la couche intermédiaire"
assert type(batch_size) == int, "batch doit être un entier"
assert type(epochs) == int, "epochs doit être entier"


print("✅ Aucun problème pour les entrées")


# ------------ Téléchargement des données  ------------ 

train_dir = r"C:\Users\23mat\.kaggle\chest_xray_data\archive\chest_xray\train" #chemin du dossier
categories = ["NORMAL", "PNEUMONIA"]

features = []
targets = []

for label, category in enumerate(categories):
    
    path = os.path.join(train_dir, category, "*.jpeg")  # ouvre toutes les images .jpeg de la category
    
    for file in glob.glob(path):
        try:
            img = Image.open(file).convert('L')  # converti en niveaux de gris
            img = img.resize((nbpix,nbpix))          # redimensionne
            features.append(np.array(img))
            if label == 0:
                targets.append(np.array([1,0])) #cas NORMAL
            else:
                targets.append(np.array([0,1])) #cas PNEUMONIA
      
        except:
             print("""/!\\ Problème dans l'importantion des image /!\\""")


features = np.array(features) / 255.0 # Normalisation
targets = np.array(targets)

indices = np.random.permutation(len(features)) # On mélange
features = features[indices]
targets = targets[indices]

print("✅ Téléchargement effectué")


# ------------ Affichage d'images pour l'exemple ------------ 


def affichage(nb):

    plt.imshow(features[nb], cmap='gray')
    plt.title("NORMAL" if targets[nb][0]==1 else "PNEUMONIA")
    plt.axis('off')
    plt.show()
    
affichage(100) # NORMAL
affichage(103) # PNEUMONIA


# ------------ Implémentation de la structure et du modèle ------------ 

# Entrées du modèle

x = tf.placeholder(tf.float32,(None, nbpix, nbpix, nbcol), name = 'Image')
y = tf.placeholder(tf.float32, (None,2), name = 'Targets')


# Filtres pour la convolution

def create_conv(prev,dimfiltre,nbfiltre):
    """
        Permet de créer les filtres
    """
    
    W_filters = tf.Variable(tf.truncated_normal(shape=(dimfiltre,dimfiltre,int(prev.get_shape()[-1]),nbfiltre)))
    b_filters = tf.Variable(tf.zeros(shape=(nbfiltre)))

    conv = tf.nn.conv2d(prev, W_filters, strides=[1,1,1,1], padding='SAME') + b_filters
    # Srides correspond à comment on se décale entre les filtres 
    # et padding dit si l'on peut déborder de l'image ou non : SAME oui VALID non

    conv = tf.nn.relu(conv) #Relu est la fonction d'activation (permet d'avoir un réseau non linéaire comme l'ajout de couches intermédiaires)
    conv = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # Implémentation du pooling
 
    return conv

conv = create_conv(x,8,32)
conv = create_conv(conv,5,64)
conv = create_conv(conv,5,64)
conv = create_conv(conv,5,128)
conv = create_conv(conv,5,256)

flat = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # On vectorise sur une ligne

# Implémentation du réseau de neuronne (couche 1)

w1 = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]),nb_neuronne_couche_int)))
b1 = tf.Variable(tf.zeros(shape=(nb_neuronne_couche_int)))

fc1 = tf.matmul(flat,w1) + b1 # correspond au produit matriciel qui renvoie les sorties de la première couche
fc1 = tf.nn.relu(fc1) #On active

# Implémentation du réseau de neuronne (couche 2)

w2 = tf.Variable(tf.truncated_normal(shape=(nb_neuronne_couche_int,2)))
b2 = tf.Variable(tf.zeros(shape=(2)))

logits = tf.matmul(fc1,w2) + b2

softmax = tf.nn.softmax(logits)


# ------------ Erreur ------------ 

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss_operation)

# ------------ Ajout artificiel d'images par transformation ------------ 


def augment_np(img_np):
    # img_np : numpy array (150,150)

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

    # Toujours revenir à 150×150
    pil = pil.resize((nbpix, nbpix))

    # Retour numpy normalisé entre 0 et 1
    img_np = np.array(pil).astype("float32") / 255.0

    return img_np.reshape(nbpix, nbpix, 1)



# ------------ Descente du gradient ------------ 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

accs = []

for epoch in range(epochs):
    
    print ("Epoch ", epoch, "/", epochs, " starting...")
    
    # On mélange 
    
    index = np.arange(len(features))
    np.random.shuffle(index)
    features = features[index]
    targets = targets[index]
    
    for b in range(0, len(features), batch_size):
        
        batch = features[b:b+batch_size]
        batch_aug = np.array([augment_np(imge) for imge in batch])
        acc = sess.run(train_op,feed_dict={x:batch_aug,y:targets[b:b+batch_size]})
        accs.append(acc)
        
        
# ------------ TEST ------------ 


test_dir = r"C:\Users\23mat\.kaggle\chest_xray_data\archive\chest_xray\test"
test_features = []
test_targets = []

for label, category in enumerate(categories):
    path = os.path.join(test_dir, category, "*.jpeg")
    for file in glob.glob(path):
        try:
            img = Image.open(file).convert("L")
            img = img.resize((nbpix, nbpix))
            arr = np.array(img).astype("float32") / 255.0
            test_features.append(arr.reshape(nbpix, nbpix, 1))
            test_targets.append(np.array([1,0]) if label == 0 else np.array([0,1]))
        except:
            print("Erreur test :", file)

test_features = np.array(test_features)
test_targets = np.array(test_targets)

print("📁 Test chargé :", len(test_features), "images")

# -----------------------------
#  Calcul de l'accuracy sur TEST
# -----------------------------

feed_test = {x: test_features, y: test_targets}
test_acc, test_loss = sess.run([accuracy, loss_operation], feed_dict=feed_test)

print("\n==============================")
print("   📌 RÉSULTATS SUR LE TEST   ")
print("==============================")
print("🔹 Test Accuracy :", round(test_acc*100, 2), "%")
print("==============================\n")


        
















