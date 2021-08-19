import numpy as np
import featureExtr
import preprocessing

xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xtrain.npy")
xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xtest.npy")
ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain.npy")
ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest.npy")

xtrain, yftrain = preprocessing.spWin(xtrain, window=1000, y=ytrain)
xtest, yftest = preprocessing.spWin(xtest, window=1000, y=ytest)

for i in range(len(xtrain)):
	xtrain[i,:,:] = preprocessing.sgnStd(xtrain[i,:,:])

for i in range(len(xtest)):
	xtest[i,:,:] = preprocessing.sgnStd(xtest[i,:,:])

xftrain = featureExtr.spectrumChn(xtrain, fs = 1000, freq = None, nfft = None)
xftest = featureExtr.spectrumChn(xtest, fs = 1000, freq = None, nfft = None)

print(xftrain)
print(xftest)

np.save(r"D:\TheodorRusnac\luiza_scripts\xftrain.npy",xftrain)
np.save(r"D:\TheodorRusnac\luiza_scripts\xftest.npy",xftest)
np.save(r"D:\TheodorRusnac\luiza_scripts\yftrain.npy",yftrain)
np.save(r"D:\TheodorRusnac\luiza_scripts\yftest.npy",yftest)
