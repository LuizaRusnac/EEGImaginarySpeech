import numpy as np
import pca
import preprocessing
import featureExtr
import machineLearning

xtrain = np.load(r"xtrain_1000.npy")
xtest = np.load(r"xtest_1000.npy")
comp_train = np.load(r"xtrain_pcaComp.npy")
comp_tst = np.load(r"xtest_pcaComp.npy")
ytrain = np.load(r"ytrain_1000.npy")
ytest = np.load(r"ytest_1000.npy")

epochs = 40;

m = np.zeros((epochs,1))
mt = np.zeros((epochs,1))
ix = np.zeros((epochs,1))
max_acc = 0.23

m[0] = 0.23
ix[0] = 5

# xtrain = preprocessing.featureNorm(xtrain)
# xtest = preprocessing.featureNorm(xtest)

rnd = np.random.permutation(len(xtrain))
nr_tr = int(0.5*len(xtrain))
xtr = xtrain[rnd[:nr_tr]]
ytr = ytrain[rnd[:nr_tr]]
xval = xtrain[rnd[nr_tr:]]
yval = ytrain[rnd[nr_tr:]]
comp_tr = comp_train[rnd[:nr_tr]]
comp_val = comp_train[rnd[nr_tr:]]

dimtr = xtr.shape
dimval = xval.shape
dimtst = xtest.shape

# xtr,ytr = preprocessing.spWin(xtr, 500, y=ytr)
# xval,yval = preprocessing.spWin(xval, 500, y=yval)
# xtest,ytest = preprocessing.spWin(xtest, 500, y=ytest)

xtr, minim, maxim = preprocessing.featureNorm(xtr, flag=1)
xval = preprocessing.featureNorm(xval, minim, maxim)
xtest = preprocessing.featureNorm(xtest, minim, maxim)

print(xtr.shape)
print(xval.shape)
print(xtest.shape)

keep = [0, len(xtr[0])]

# xhattr, comp_tr = pca.allDataPca(xtr, keep = keep, flag = 0)
# xhatval, comp_val = pca.allDataPca(xval, keep = keep, flag = 0)
# xhattest, comp_tst = pca.allDataPca(xtest, keep = keep, flag = 0)


D = np.random.normal(size=(62,1000))
xhatvalant = np.random.normal(size=(len(xval),62,1000))
xhattest = np.random.normal(size=(len(xval),62,1000))

for it in range(epochs):
	print("You are at epoch %d/%d"%(it+1,epochs))

	comp_tr = comp_tr+D
	comp_val = comp_val+D
	comp_tst = comp_tst+D

	xhattr = pca.allDataTransf(xtr, comp_tr, keep)
	xhatval = pca.allDataTransf(xval, comp_val, keep)

	# xftrain = featureExtr.spectrumChn(xhattr, fs = 1000, freq = None, nfft = 2000)
	# xfval = featureExtr.spectrumChn(xhatval, fs = 1000, freq = None, nfft = 2000)
	# print(xftrain.shape)

	xftrain = preprocessing.mat3d2mat2d(xhattr)
	xfval = preprocessing.mat3d2mat2d(xhatval)
	yftrain = np.ravel(ytr)
	yfval = np.ravel(yval)

	score, predict = machineLearning.knn(xftrain,yftrain,xfval,yfval, kval = [3], flag = 1)

	print("True label:")
	print(yfval)
	print("Predict label:")
	print(predict)

	m[it] = score
	print(D)

	#xftrain = preprocessing.mat2d2mat3d(xftrain,dimtr[1],dimtr[2])

	for i in range(len(predict)):
		if predict[i]==yfval[i]:
			D = D + 0.01 * xhatval[i,:,:]*(1-score)#*np.random.normal(size=(62,1000))#*(xhattstant[i,:,:])*xhattstant2[i,:,:]#np.random.normal(size=(62,1000))#*(1-xhattst[i,:,:])
		else:
			D = D - 0.01 * (xhatval[i,:,:])*(1-score)#*np.random.normal(size=(62,1000))#*(xhattstant[i,:,:])*xhattstant2[i,:,:]#np.random.normal(size=(62,1000))#*(1-score)#*(1-xhattst[i,:,:])


	D = preprocessing.featureNormRange(D,rng=[-10,10])
	#D = preprocessing.featureNorm(D)#+np.random.normal(size=(62,1000))
	Ds = np.argsort(D,axis=0)
	print(Ds)

	xhattst= pca.allDataTransf(xtest, comp_tst, keep)
	# xftest = featureExtr.spectrumChn(xhattst, fs = 1000, freq = None, nfft = 2000)
	xftest = preprocessing.mat3d2mat2d(xhattst)
	yftest = np.ravel(ytest)

	score2, predict2 = machineLearning.knn(xftrain,yftrain,xftest,yftest, kval = [3], flag = 1)
	print("Acc pe setul de test este: ")
	print(score2)
	print("\n")
	mt[it] = score2


print(m)
print("\n")
print(mt)
print("\n")
print(D)


