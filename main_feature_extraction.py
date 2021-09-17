import numpy as np
import featureExtr
import preprocessing
from scipy.ndimage import gaussian_filter

xtrain = np.load(r"Xtrain.npy")
xtest = np.load(r"Xtest.npy")
ytrain = np.load(r"ytrain.npy")
ytest = np.load(r"ytest.npy")

# xtrain = np.load(r"xfdbtrain.npy")
# xtest = np.load(r"xfdbtest.npy")
# ytrain = np.load(r"ytrain.npy")
# ytest = np.load(r"ytest.npy")

xtrain, yftrain = preprocessing.spWin(xtrain, window=1000, y=ytrain)
xtest, yftest = preprocessing.spWin(xtest, window=1000, y=ytest)


# for i in range(len(xtrain)):
# 	xtrain[i,:,:] = preprocessing.sgnStd(xtrain[i,:,:])

# for i in range(len(xtest)):
# 	xtest[i,:,:] = preprocessing.sgnStd(xtest[i,:,:])


xftrain = featureExtr.spectrumChn(xtrain, fs = 1000, freq = [0, 500], nfft = None)
xftest = featureExtr.spectrumChn(xtest, fs = 1000, freq = [0, 500], nfft = None)

print(xftrain)
print(xftest)

# xctrain = np.zeros((xftrain.shape[0],xftrain.shape[1],xftrain.shape[1]))
# for i in range(len(xftrain)):
# 	xf = gaussian_filter(xftrain[i,:,:], sigma=1)
# 	xctrain[i,:,:] = np.corrcoef(xf)

# xctest = np.zeros((xftest.shape[0],xftest.shape[1],xftest.shape[1]))
# for i in range(len(xtest)):
# 	xf = gaussian_filter(xftest[i,:,:], sigma=1)
# 	xctest[i,:,:] = np.corrcoef(xf)


np.save(r"xftrain.npy",xftrain)
np.save(r"xftest.npy",xftest)
np.save(r"yftrain.npy",yftrain)
np.save(r"yftest.npy",yftest)
