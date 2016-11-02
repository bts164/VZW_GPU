import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from struct import *
from array import array

f = open("psulion.yuv", "rb")
y = f.read(1920*1080)
cb = f.read(int(1920*1080/4))
cr = f.read(int(1920*1080/4))
#close(f)

Y = np.zeros((1080, 1920), dtype=float)
for j in range(0, 1920):
    for i in range(0, 1080):
        Y[i][j] = float(y[i*1920+j])

CB = np.zeros((int(1080/2), int(1920/2)), dtype=float)
CR = np.zeros((int(1080/2), int(1920/2)), dtype=float)
for j in range(0, int(1920/2)):
    for i in range(0, int(1080/2)):
        CB[i][j] = float(cb[i*int(1920/2)+j])-128
        CR[i][j] = float(cr[i*int(1920/2)+j])-128

R = np.zeros((1080, 1920), dtype=float)
G = np.zeros((1080, 1920), dtype=float)
B = np.zeros((1080, 1920), dtype=float)
for j in range(0, 1920):
    for i in range(0, 1080):
        iIdx = int(i/2)
        jIdx = int(j/2)
        R[i][j] = Y[i][j] + 1.370705 * CR[iIdx][jIdx]
        G[i][j] = Y[i][j] - 0.698001 * CR[iIdx][jIdx] - 0.337633 * CB[iIdx][jIdx]
        B[i][j] = Y[i][j] + 1.732446 * CB[iIdx][jIdx]

#for j in range(0, 1920):
#    for i in range(0, 1080):
#        LR = 
plt.imshow(B, cmap='gray')
plt.show()

