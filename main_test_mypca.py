import numpy as np
import matplotlib.pyplot as plt
from EEGPlot import eegGR
from numpy.linalg import eig
from sklearn.decomposition import PCA
from preprocessing import sgnStd

xtrain = np.load(r"Xtrain.npy")
xtrain = xtrain[100]
mean = xtrain.mean(axis=0)

xtrain = xtrain - mean
# xtrain = sgnStd(xtrain)
c = np.cov(xtrain.T)
values, vectors = eig(c)

pca = PCA()
pca.fit(xtrain)
# print(pca.components_.shape)
# vectors = np.abs(vectors)
print("Eig vectors: ")
print(vectors.T[:62,:])
print("Eig abs(vectors): ")
print(np.abs(vectors.T[:62,:]))
print("Eig real(vectors): ")
print(np.real(vectors.T[:62,:]))
print("Computed pca comp: ")
print(pca.components_)

eegGR(xtrain[:,:],[10],"pca_sgn.png",space = 'minim')

#eegGR(vectors[:,:4000],[10],"pca_linii.png")
eegGR(vectors.T[:,:500],[10],"pca_coloane.png",space = 'minim')
eegGR(pca.components_[:,:500],[10],"pca_sklearn_comp.png",space = 'minim')

# vectors[:,:2] = 0

print(vectors)

# xhat = 
vectors = np.real(vectors)
vectors[:, :2] = 0
saprox = np.dot(xtrain, vectors.T)
print(saprox.shape)
eegGR(saprox,[10],"pca_transf_sgn.png",space = 'minim')
saprox = np.dot(saprox, vectors)


# print(saprox)
eegGR(vectors.T[:,:500],[10],"pca_coloane.png",space = 'minim')
eegGR(saprox,[10],"pca_reconstr_sgn.png",space = 'minim')

pca.components_[:2,:] = 0
pcaTransf = np.dot(xtrain, pca.components_.T)
xhat = np.dot(pcaTransf, pca.components_)
# keep = [2, 62]
# pcaTransf = np.dot(xtrain, pca.components_[keep[0]:keep[1],:].T)
# xhat = np.dot(pcaTransf, pca.components_[keep[0]:keep[1],:])
print(pcaTransf.shape)
eegGR(xhat,[10],"pca_reconstr_sklearn.png",space = 'minim')
