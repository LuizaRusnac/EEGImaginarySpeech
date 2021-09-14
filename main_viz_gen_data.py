import numpy as np
from EEGPlot import eegGR

gen = np.load("generated_pca.npy")
ygen = np.load("generate_pca_labels.npy")

x = np.load("xtrain.npy")
# y = np.load("ytrain_pca.npy")

print(x.shape)
# print(gen.shape)
# eegGR(gen[100], ch = [10], name = 'sgn_gen_10ch.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)
eegGR(x[100,:,:500], ch = [10], name = 'sgn_real_10ch.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)
