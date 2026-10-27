# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def flou (L):
    
    l = len(L[0])
    h = len(L)
    
    M = np.zeros((h, l, 3))
    
    for c in range (l):
        for j in range (h):
            
            if c == 0 :
                
                if j == 0 :
                    M[j][c] = (1/4)*L[j][c] + (1/4)*L[j+1][c] + (1/4)*L[j+1][c+1] + (1/4)*L[j][c+1]
                
                elif j == h-1:
                    M[j][c] = (1/4)*L[j][c] + (1/4)*L[j-1][c] + (1/4)*L[j-1][c+1] + (1/4)*L[j][c+1]

                else:
                    M[j][c] = (1/6)*L[j][c] + (1/6)*L[j+1][c] + (1/6)*L[j-1][c] + (1/6)*L[j+1][c+1] + (1/6)*L[j-1][c+1] + (1/6)*L[j][c+1]

            elif c==l-1 :
                
                if j == 0 :
                    M[j][c] = (1/4)*L[j][c] + (1/4)*L[j+1][c] + (1/4)*L[j+1][c-1] + (1/4)*L[j][c-1]
                
                elif j == h-1:
                    M[j][c] = (1/4)*L[j][c] + (1/4)*L[j-1][c] + (1/4)*L[j-1][c-1] + (1/4)*L[j][c-1]

                else:
                    M[j][c] = (1/6)*L[j][c] + (1/6)*L[j+1][c] + (1/6)*L[j-1][c] + (1/6)*L[j+1][c-1] + (1/6)*L[j-1][c-1] + (1/6)*L[j][c-1]

            else:
                
                if j == 0 : 
                    M[j][c] = (1/6)*L[j][c] + (1/6)*L[j+1][c] + (1/6)*L[j][c+1] + (1/6)*L[j][c-1] + (1/6)*L[j+1][c+1] + (1/6)*L[j+1][c-1]

                elif j == h-1:
                    M[j][c] = (1/6)*L[j][c] + (1/6)*L[j-1][c] + (1/6)*L[j][c+1] + (1/6)*L[j][c-1] + (1/6)*L[j-1][c+1] + (1/6)*L[j-1][c-1]

                else:
                    M[j][c] = (1/9)*L[j][c] + (1/9)*L[j+1][c] + (1/9)*L[j-1][c] + (1/9)*L[j][c+1] + (1/9)*L[j-1][c+1] + (1/9)*L[j+1][c+1] + (1/9)*L[j][c-1] + (1/9)*L[j+1][c-1] + (1/9)*L[j-1][c-1]


    return M.astype(np.uint8)

def contour(L):
    h = len(L)
    l = len(L[0])

    M = np.zeros((h, l))

    K = np.array([[ 0.25, 0,  -0.25],        #détecte les frontières verticales
                  [0.50,  0, -0.5],
                  [ 0.25, 0,  -0.25]])
    
    # K = np.array([[ 0.25, 0.5, 0.25],        #détecte les frontières horizontales
    #               [0, 0, 0],
    #               [ -0.25, -0.5, -0.25]])

    for j in range(h):
        for c in range(l):

            # Coin haut-gauche
            if j == 0 and c == 0:
                M[j][c] = K[1][1]*L[j][c] + K[1][2]*L[j][c+1] + K[2][1]*L[j+1][c] + K[2][2]*L[j+1][c+1]

            # Coin haut-droit
            elif j == 0 and c == l-1:
                M[j][c] = K[1][1]*L[j][c] + K[1][0]*L[j][c-1] + K[2][1]*L[j+1][c] + K[2][0]*L[j+1][c-1]

            # Coin bas-gauche
            elif j == h-1 and c == 0:
                M[j][c] = K[1][1]*L[j][c] + K[1][2]*L[j][c+1] + K[0][1]*L[j-1][c] + K[0][2]*L[j-1][c+1]

            # Coin bas-droit
            elif j == h-1 and c == l-1:
                M[j][c] = K[1][1]*L[j][c] + K[1][0]*L[j][c-1] + K[0][1]*L[j-1][c] + K[0][0]*L[j-1][c-1]

            # Bord gauche (milieu)
            elif c == 0:
                M[j][c] = K[1][1]*L[j][c] + K[1][2]*L[j][c+1] + K[0][1]*L[j-1][c] + K[0][2]*L[j-1][c+1] + K[2][1]*L[j+1][c] + K[2][2]*L[j+1][c+1]

            # Bord droit (milieu)
            elif c == l-1:
                M[j][c] = K[1][1]*L[j][c] + K[1][0]*L[j][c-1] + K[0][1]*L[j-1][c] + K[0][0]*L[j-1][c-1] + K[2][1]*L[j+1][c] + K[2][0]*L[j+1][c-1]

            # Bord haut (milieu)
            elif j == 0:
                M[j][c] = K[1][1]*L[j][c] + K[1][0]*L[j][c-1] + K[1][2]*L[j][c+1] + K[2][0]*L[j+1][c-1] + K[2][1]*L[j+1][c] + K[2][2]*L[j+1][c+1]

            # Bord bas (milieu)
            elif j == h-1:
                M[j][c] = K[1][1]*L[j][c] + K[1][0]*L[j][c-1] + K[1][2]*L[j][c+1] + K[0][0]*L[j-1][c-1] + K[0][1]*L[j-1][c] + K[0][2]*L[j-1][c+1]

            # Intérieur
            else:
                M[j][c] = (
                    K[0][0]*L[j-1][c-1] + K[0][1]*L[j-1][c] + K[0][2]*L[j-1][c+1] +
                    K[1][0]*L[j][c-1]   + K[1][1]*L[j][c]   + K[1][2]*L[j][c+1] +
                    K[2][0]*L[j+1][c-1] + K[2][1]*L[j+1][c] + K[2][2]*L[j+1][c+1]
                )

    M = np.abs(M)
    M = M / M.max() * 255
    return M.astype(np.uint8)



plt.figure()
I = Image.open("cerisiersfleurs.png")
I1 = np.array(I.convert("RGB"))
plt.imshow(I1)
plt.axis("off")
plt.show()

plt.figure()
M = flou(flou(flou(flou(flou(I1)))))
plt.imshow(M)
plt.axis("off")
plt.show()

plt.figure()
I2 = np.array(I.convert("L"))
plt.imshow(I2, cmap="gray")
plt.axis("off")
plt.show()

plt.figure()
N = contour(I2)
plt.imshow(N, cmap="gray")
plt.axis("off")
plt.show()
