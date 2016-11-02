import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import struct as struct
from array import array

data_type = dtype=np.dtype('u1')
bindata = np.fromfile("rgbdata.dat", data_type)
R = np.zeros((1080, 1920), dtype=data_type)
G = np.zeros((1080, 1920), dtype=data_type)
B = np.zeros((1080, 1920), dtype=data_type)
for j in range(0, 1920):
    for i in range(0, 1080):
        R[i][j] = bindata[0*1920*1080 + i*1920+j]
        G[i][j] = bindata[1*1920*1080 + i*1920+j]
        B[i][j] = bindata[2*1920*1080 + i*1920+j]

plt.subplot(131)
plt.imshow(R, cmap='gray', interpolation='nearest')
plt.title("Red")
plt.subplot(132)
plt.imshow(G, cmap='gray', interpolation='nearest')
plt.title("Green")
plt.subplot(133)
plt.imshow(B, cmap='gray', interpolation='nearest')
plt.title("Blue")

plt.show()
