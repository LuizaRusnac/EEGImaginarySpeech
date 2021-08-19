import numpy as np
import machineLearning
import preprocessing
import log
import KOtasks

method = 'PCA'
task = 'nasal'

if method == 'PCA':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xftest_pca.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest_pca.npy")

if method == 'No Filter':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xftest.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\yftrain.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\yftest.npy")

# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)

xtrain = preprocessing.mat3d2mat2d(xtrain)
xtest = preprocessing.mat3d2mat2d(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)

if task!= 'no task':
	ytrain = KOtasks.task(ytrain, task)
	ytest = KOtasks.task(ytest, task)

score = machineLearning.knn(xtrain,ytrain,xtest,ytest, kval = [1, 50, 2], flag = 1)

m = score.max()
ix = (score.argmax()*2)+1
text = "%s 1s standardized signal before feature extraction \n \
Feature extraction: Spectrum all frequencies \n \
With KNN, Task: %s \n \
kval from 1 to 50 \n \
max val acc: %.4f, k max val: %d"%(method,task,m,ix)

log.wlog("log_server.txt",text = text, flag = 1)