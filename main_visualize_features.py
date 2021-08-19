import numpy as np

import matplotlib.pyplot as plt

xtr = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")


plt.figure()
plt.imshow(xtr[0][:,:100])
plt.savefig("feature_image.png")
