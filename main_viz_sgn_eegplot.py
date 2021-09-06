import numpy as np
import preprocessing
import EEGPlot

x = np.load(r"D:\TheodorRusnac\luiza_scripts\Xtrain.npy")
xpca = np.load("Xtrain_pca.npy")
print(xpca)

x = preprocessing.spWin(x, 1000)

EEGPlot.eegGR(x[0], ch = [10], name = 'sgn_raw.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)
EEGPlot.eegGR(xpca[0], ch = [10], name = 'sgn_pca.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)