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

# D = np.random.normal(size=(62,1))
epochs = 20;

m = np.zeros((epochs,1))
ix = np.zeros((epochs,1))
max_acc = 0.23

m[0] = 0.23
ix[0] = 5

xtrain = preprocessing.featureNorm(xtrain)
xtest = preprocessing.featureNorm(xtest)
# for rec,i in zip(xtrain, range(len(xtrain))):
# # 	# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain)))
# 	xtrain[i,:,:] = preprocessing.sgnStd(rec)

# for rec,i in zip(xtest, range(len(xtest))):
# # 	# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain)))
# 	xtest[i,:,:] = preprocessing.sgnStd(rec)

# D = np.zeros((62,1000))
D = np.random.normal(size=(62,1000))
xhattstant = np.random.normal(size=(len(xtest),62,1000))
xhattstant2 = np.random.normal(size=(len(xtest),62,1000))
for it in range(epochs):
	print("You are at epoch %d/%d"%(it+1,epochs))

	# comp_tr = 1/(1+np.exp(-comp_tr))
	# comp_tst = 1/(1+np.exp(-comp_tst))

	# for i in range(len(comp_tr)):
	# 	if(comp_tr[i,0,0]==0):
	# 		comp_tr[i,2:,:] = 1/np.exp(-(comp_tr[i,2:,:] + D[2:,:]))
	# 	else:
	# 		comp_tr[i,:,:] = 1/np.exp(-(comp_tr[i,:,:] + D))

	# for i in range(len(comp_tst)):
	# 	if(comp_tst[i,0,0]==0):
	# 		comp_tst[i,2:,:] = 1/np.exp(-(comp_tst[i,2:,:] + D[2:,:]))
	# 	else:
	# 		comp_tst[i,:,:] = 1/np.exp(-(comp_tst[i,:,:] + D))
	comp_tr = comp_tr+D
	comp_tst = comp_tst+D

	# comp_tr = preprocessing.featureNorm(comp_tr)
	# comp_tst = preprocessing.featureNorm(comp_tst)
	# xhattr = np.zeros((xtrain.shape))
	
	# print(comp_tr)
	xhattr = np.zeros((xtrain.shape))
	for rec,i in zip(xtrain, range(len(xtrain))):
		# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain)))
		xhattr[i,:,:] = pca.pcaTransform(rec, comp_tr[i,:,:], keep = [0, len(rec)])

	xhattst = np.zeros((xtest.shape))


	for rec,i in zip(xtest, range(len(xtest))):
		# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtest)))
		xhattst[i,:,:] = pca.pcaTransform(rec, comp_tst[i,:,:], keep = [0, len(rec)])

	# xftrain = featureExtr.spectrumChn(xhattr, fs = 1000, freq = [0, 500], nfft = None)
	# xftest = featureExtr.spectrumChn(xhattst, fs = 1000, freq = [0, 500], nfft = None)
	# xftrain = np.zeros((xhattr.shape))
	# for i in range(len(xhattr)):
	# 	xftrain[i,:,:] = 1/(1+np.exp(-xhattr[i,:,:]))

	# xftest = np.zeros((xhattst.shape))
	# for i in range(len(xhattst)):
	# 	xftest[i,:,:] = 1/(1+np.exp(-xhattst[i,:,:]))



	xftrain = xhattr
	xftest = xhattst

	# print("Xftest is:")
	# print(xftrain.shape)
	# print("\n")

	xftrain = preprocessing.mat3d2mat2d(xftrain)
	xftest = preprocessing.mat3d2mat2d(xftest)
	yftrain = np.ravel(ytrain)
	yftest = np.ravel(ytest)

	score, predict = machineLearning.knn(xftrain,yftrain,xftest,yftest, kval = [3], flag = 1)

	print(predict)
	m[it] = score
	# ix[it] = (score.argmax()*2)+1
	print(D)

	# if it>0:
	# 	if(m[it]>m[it-1]):
	# 		# D = D + np.random.normal(0,0.1,size=(62,1000))
	# 		# D = D + D * np.sum(comp_tr,axis=0) * 0.01
	# 		D = D + X*()
	# 	else:
	# 		# D = D - np.random.normal(0,0.1,size=(62,1000))
	# 		# D = D - D * np.sum(comp_tr,axis=0) * 0.01

	# else:
	# 	D = D - np.random.normal(0,0.2,size=(62,1))
	# 	# D = np.random.normal(size=(62,1000))
	# derivplus = np.zeros(len(xftest[0]),len(xftest[0][0]))
	# derivminus = np.zeros(len(xftest[0]),len(xftest[0][0]))
	for i in range(len(predict)):
		if predict[i]==yftest[i]:
			# D = D + 0.000001*(xtest[i,:,:]@(comp_tst[i,:,:]+D).T)@(comp_tst[i,:,:]+D)
			D = D + 0.01 * xhattst[i,:,:]*(1-score)*comp_tst[i,:,:]#*(xhattstant[i,:,:])*xhattstant2[i,:,:]#np.random.normal(size=(62,1000))#*(1-xhattst[i,:,:])
		else:
			# D = D - 0.000001*(xtest[i,:,:]@(comp_tst[i,:,:]+D).T)@(comp_tst[i,:,:]+D)
			D = D - 0.01 * (xhattst[i,:,:])*(1-score)*comp_tst[i,:,:]#*(xhattstant[i,:,:])*xhattstant2[i,:,:]#np.random.normal(size=(62,1000))#*(1-score)#*(1-xhattst[i,:,:])


	D = preprocessing.featureNorm(D)
	Ds = np.argsort(D,axis=0)
	print(Ds)

	xhattstant2 = xhattstant
	xhattstant = xhattst

	xhattrs = np.zeros((xtrain.shape))
	for rec,i in zip(xtrain, range(len(xtrain))):
		# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtrain)))
		xhattrs[i,:,:] = pca.pcaTransform(rec, comp_tr[i,Ds[:,0],:]+D[Ds[:,0],:], keep = [57, 62])

	xhattsts = np.zeros((xtest.shape))
	for rec,i in zip(xtest, range(len(xtest))):
		# print("PCA reconstruct: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(xtest)))
		xhattsts[i,:,:] = pca.pcaTransform(rec, comp_tst[i,Ds[:,0],:]+D[Ds[:,0],:], keep = [57, 62])

	xhattrs = preprocessing.mat3d2mat2d(xhattrs)
	xhattsts = preprocessing.mat3d2mat2d(xhattsts)

	score, predict = machineLearning.knn(xhattrs,yftrain,xhattsts,yftest, kval = [3], flag = 1)
	print("Second score:")
	print(score)
	print("\n")

	# D = D.T
	# print(D.shape)
	# D = D/len(predict)
	
	# D = D  0.01*np.sum(xftest[predict!=yftest]@(comp_tst[predict!=yftest]+D).T@(comp_tst[predict!=yftest]+D),axis=0)



print(m)
print(D)


