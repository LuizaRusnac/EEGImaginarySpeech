import numpy as np
import pca
import preprocessing
import featureExtr
import machineLearning

xtrain = np.load(r"xtrain_1000.npy")
xtest = np.load(r"xtest_1000.npy")
comp_tr = np.load(r"xtrain_pcaComp.npy")
comp_tst = np.load(r"xtest_pcaComp.npy")
ytrain = np.load(r"ytrain_1000.npy")
ytest = np.load(r"ytest_1000.npy")

D = np.random.normal(size=(62,1))
epochs = 10;

m = np.zeros((epochs+1,1))
ix = np.zeros((epochs+1,1))
max_acc = 0.23

m[0] = 0.23
ix[0] = 5

for it in range(10):
	for i in range(len(comp_tr)):
		if(comp_tr[i,0,0]==0):
			comp_tr[i,2:,:] = comp_tr[i,2:,:] + D[2:]
		else:
			comp_tr[i,:,:] = comp_tr[i,:,:] + D

	for i in range(len(comp_tst)):
		if(comp_tst[i,0,0]==0):
			comp_tst[i,2:,:] = comp_tst[i,2:,:] + D[2:]
		else:
			comp_tst[i,:,:] = comp_tst[i,:,:] + D

	xhattr = np.zeros((xtrain.shape))
	for rec,i in zip(xtrain, range(len(xtrain))):
		print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain)))
		xhattr[i,:,:] = pca.pcaTransform(rec, comp_tr[i,:,:], keep = [0, len(rec)])

	xhattst = np.zeros((xtest.shape))
	for rec,i in zip(xtest, range(len(xtest))):
		print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtest)))
		xhattst[i,:,:] = pca.pcaTransform(rec, comp_tst[i,:,:], keep = [0, len(rec)])

	xftrain = featureExtr.spectrumChn(xhattr, fs = 1000, freq = [0, 500], nfft = None)
	xftest = featureExtr.spectrumChn(xhattst, fs = 1000, freq = [0, 500], nfft = None)

	xftrain = preprocessing.mat3d2mat2d(xftrain)
	xftest = preprocessing.mat3d2mat2d(xftest)
	yftrain = np.ravel(ytrain)
	yftest = np.ravel(ytest)

	score = machineLearning.knn(xftrain,yftrain,xftest,yftest, kval = [1, 50, 2], flag = 1)

	m[it+1] = score.max()
	ix[it+1] = (score.argmax()*2)+1

	if(m[it+1]>m[it]):
		D = D + np.random.normal(size=(62,1))
	else:
		D = D - np.random.normal(size=(62,1))

print(m)
print(D)


