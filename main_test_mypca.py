import numpy as np
import matplotlib.pyplot as plt
from EEGPlot import eegGR
from numpy.linalg import eig
from sklearn.decomposition import PCA

xtrain = np.load(r"Xtrain.npy")
xtrain = xtrain[1]
mean = xtrain.mean(axis=0)

xtrain = xtrain - mean
c = np.cov(xtrain.T)
values, vectors = eig(c)

pca = PCA()
pca.fit(xtrain)

vectors = real(vectors)
print(vectors)

eegGR(xtrain[:,:4000],[10],"pca_sgn.png",space = 'maxim')

#eegGR(vectors[:,:4000],[10],"pca_linii.png")
eegGR(vectors.T[:,:4000],[10],"pca_coloane.png",space = 'maxim')
eegGR(pca.components_,[10],"pca_sklearn_comp.png",space = 'maxim')
