import numpy as np
import machineLearning
import preprocessing
import log
import KOtasks

method = 'No Filter'
task = 'no task'

if method == 'PCA':
	xtrain = np.load(r"xftrain_pca.npy")
	xtest = np.load(r"xftest_pca.npy")
	ytrain = np.load(r"yftrain_pca.npy")
	ytest = np.load(r"yftest_pca.npy")
	# gen = np.load('generated_pca.npy')
	# ygen = np.load('generate_pca_labels.npy')
	# ygen = np.reshape(ygen,(len(ygen),1))
	# print(xtrain.shape)

if method == 'No Filter':
	xtrain = np.load(r"xtrain.npy")
	xtest = np.load(r"xtest.npy")
	ytrain = np.load(r"ytrain.npy")
	ytest = np.load(r"ytest.npy")
	
print(xtrain.shape)
# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)
# xtrain = xtrain[:,2:,:]
# xtest = xtest[:,2:,:]
xtrain, ytrain = preprocessing.spWin(xtrain, 1000, ytrain)
xtest, ytest = preprocessing.spWin(xtest, 1000, ytest)
# print(xtrain.shape)
# xtrain = np.resize(xtrain,(xtrain.shape[0],60,500))
# xtest = np.resize(xtest,(xtest.shape[0],60,500))
xctrain = np.zeros((xtrain.shape[0],xtrain.shape[1],xtrain.shape[1]))
for i in range(len(xtrain)):
	xctrain[i,:,:] = np.cov(xtrain[i,:,:])

xctest = np.zeros((xtest.shape[0],xtest.shape[1],xtest.shape[1]))
for i in range(len(xtest)):
	xctest[i,:,:] = np.cov(xtest[i,:,:])

# xtrain = np.resize(xctrain,(xtrain.shape[0],60,60))
# xtest = np.resize(xctest,(xtest.shape[0],60,60))
# xtrain, minim, maxim = preprocessing.featureStd(xtrain, flag = 1)
# xtest = preprocessing.featureStd(xtest, minim, maxim)
# xtrain = np.concatenate((xtrain,gen))
xtrain = preprocessing.mat3d2mat2d(xctrain)
xtest = preprocessing.mat3d2mat2d(xctest)
# ytrain = np.concatenate((ytrain,ygen))
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)

if task!= 'no task':
	ytrain = KOtasks.task(ytrain, task)
	ytest = KOtasks.task(ytest, task)

score = machineLearning.knn(xtrain,ytrain,xtest,ytest, kval = [1, 50, 2], flag = 1)

m = score.max()
ix = (score.argmax()*2)+1
text = "%s 4s NO standardized signal before feature extraction \n \
Feature extraction: Time series \n \
With KNN, Task: %s \n \
Cov => 62x62 \n\
kval from 1 to 50 \n \
max val acc: %.4f, k max val: %d"%(method,task,m,ix)

log.wlog("log_server_knn.txt",text = text, flag = 1)