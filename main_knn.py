import numpy as np
import machineLearning
import preprocessing
import log
import KOtasks
from preprocessing import sgnStd, sgnNorm, featureStd
import matplotlib.pyplot as plt
import numpy.matlib

method = 'PCA'
task = 'no task'

if method == 'PCA':
	xtrain = np.load(r"xftrain_pca.npy")
	xtest = np.load(r"xftest_pca.npy")
	ytrain = np.load(r"yftrain_pca.npy")
	ytest = np.load(r"yftest_pca.npy")

if method == 'No Filter':
	xtrain = np.load(r"Xtrain.npy")
	xtest = np.load(r"Xtest.npy")
	ytrain = np.load(r"ytrain.npy")
	ytest = np.load(r"ytest.npy")
	gen = np.load('generated_pca.npy')
	ygen = np.load('generate_pca_labels.npy')
	ygen = np.reshape(ygen,(len(ygen),1))
	print(gen.shape)
	
print(xtrain.shape)
# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)
# xtrain = xtrain[:,2:,:]
# xtest = xtest[:,2:,:]



# xtrain, ytrain = preprocessing.spWin(xtrain, 1000, ytrain)
# xtest, ytest = preprocessing.spWin(xtest, 1000, ytest)
# # print(xtrain.shape)
# xtrain = np.resize(xtrain,(xtrain.shape[0],60,500))
# xtest = np.resize(xtest,(xtest.shape[0],60,500))

# xctrain = np.zeros((xtrain.shape[0],xtrain.shape[1],xtrain.shape[1]))
# for i in range(len(xtrain)):
# 	xctrain[i,:,:] = np.cov(xtrain[i,:,:])

# xctest = np.zeros((xtest.shape[0],xtest.shape[1],xtest.shape[1]))
# for i in range(len(xtest)):
# 	xctest[i,:,:] = np.cov(xtest[i,:,:])

# dim = xtrain.shape

# for i in range(len(xtrain)):
# 	mean = np.reshape(xtrain[i,:,:].mean(axis=1),(dim[1],1))
# 	xtrain[i,:,:] = xtrain[i,:,:] - numpy.matlib.repmat(mean,1,dim[2])

# dim = xtest.shape
# for i in range(len(xtest)):
# 	mean = np.reshape(xtest[i,:,:].mean(axis=1),(dim[1],1))
# 	xtest[i,:,:] = xtest[i,:,:] - numpy.matlib.repmat(mean,1,dim[2])

# for x in xtrain[0]:
# 	plt.figure()
# 	plt.plot(x)
# 	plt.show()
# 	plt.close()
# xtrain, mean, std = featureStd(xtrain, flag=1)
# xtest = featureStd(xtest, mean, std)
# xtrain = np.resize(xctrain,(xtrain.shape[0],60,60))
# xtest = np.resize(xctest,(xtest.shape[0],60,60))
# xtrain, minim, maxim = preprocessing.featureStd(xtrain, flag = 1)
# xtest = preprocessing.featureStd(xtest, minim, maxim)

# xtrain = np.concatenate((xctrain,gen),axis=0)
# # print(xtrain.shape)
xtrain = preprocessing.mat3d2mat2d(xtrain)
xtest = preprocessing.mat3d2mat2d(xtest)
# ytrain = np.reshape(ytrain,(len(ytrain),1))
# ytrain = np.concatenate((ytrain,ygen))

ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)

if task!= 'no task':
	ytrain = KOtasks.task(ytrain, task)
	ytest = KOtasks.task(ytest, task)

print(xtrain.shape)
score = machineLearning.knn(xtrain,ytrain,xtest,ytest, kval = [1, 50, 2], flag = 1)

m = score.max()
ix = (score.argmax()*2)+1
text = "%s 1s Signal PCA withouth normalization/standardization signals \n \
Feature extraction: Spectrum, all freq \n \
With KNN, Task: %s \n \
kval from 1 to 50 \n \
max val acc: %.4f, k max val: %d"%(method,task,m,ix)

log.wlog("log_server_knn.txt",text = text, flag = 1)