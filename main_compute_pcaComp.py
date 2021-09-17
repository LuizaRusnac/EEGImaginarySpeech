import numpy as np
import pca
import preprocessing
import featureExtr
import machineLearning

xtrain = np.load(r"Xtrain.npy")
xtest = np.load(r"Xtest.npy")
ytrain = np.load(r"ytrain.npy")
ytest = np.load(r"ytest.npy")

xtrain0, yftrain = preprocessing.spWin(xtrain, 1000, y=ytrain)
xtest0, yftest = preprocessing.spWin(xtest, 1000, y=ytest)

comp_tr = np.zeros((xtrain0.shape))
for rec,i in zip(xtrain0, range(len(xtrain0))):
	print("PCA: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain0)))
	comp = pca.pcaComp(rec)
	if(np.corrcoef(comp[:2,:])[0,1]>0.97):
		comp[:2,:] = 0
	comp_tr[i,:,:] = pca.pcaComp(rec)

comp_tst = np.zeros((xtest0.shape))

for rec,i in zip(xtest0, range(len(xtest0))):
	print("PCA: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtest0)))
	comp = pca.pcaComp(rec)
	if(np.corrcoef(comp[:2,:])[0,1]>0.97):
		comp[:2,:] = 0
	comp_tst[i,:,:] = comp

np.save("xtrain_1000.npy",xtrain0)
np.save("xtest_1000.npy",xtest0)
np.save("ytrain_1000.npy",yftrain)
np.save("ytest_1000.npy",yftest)
np.save("xtrain_pcaComp.npy",comp_tr)
np.save("xtest_pcaComp.npy",comp_tst)