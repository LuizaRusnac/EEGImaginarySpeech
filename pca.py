from sklearn.decomposition import PCA
import numpy as np
import numpy.matlib

"""
This module contains:

pcaComp - computes the PCA components using sklearn.decomposition module

pcaTransform - transform the original matrix X into the reconstructed matrix using the desired components

allDataPca - computes the transformed matrix x of 3D. This function is used to compute sources of channels and eliminate the
	desired ones over the all records

"""


def pcaComp(X):
	"""
	This function computes the PCA components using sklearn.decomposition module

	Input data:
		X - A 2D matrix. Dimesion: [N x M], N = number of observations, M = number of features

	Output data:
		pca.components_ - the computed PCA componens of matrix X. Dimension: [min(M,N) x M]

	"""
	pca = PCA()
	pca.fit(X)

	return pca.components_

def pcaTransform(X, pcaComp, keep = [0, 2], flag = 0):
	"""
	This functions transform the original matrix X into the reconstructed matrix using the desired components

	Inut data:
		X - A 2D matrix. Dimesion: [NxM], N = number of observations, M = number of features
		pcaComp - the PCA components
		keep - the components to keep from keep[0] to keep[1]. DEFAULT = [0, 2], it keeps the first 2 components.
		flag - takes values 0, 1, 2 or 3. If flag = 0 the data will remain the same. If flag = 1, all observations of the
		data will be moved to 0 mean. If flag = 2, data will be standardized over features. If flag = 3, data will be
		normalized over features.

	Output data:
		Xhat - the aproximate X data after eliminating the desired components

	"""

	dim = X.shape

	if flag == 1:
		mean = np.reshape(X.mean(axis=0),(1,dim[1]))
		X = (X - numpy.matlib.repmat(mean,dim[0],1))
	elif flag==2:
		mean = np.reshape(X.mean(axis=0),(1,dim[1]))
		std = np.reshape(X.std(axis=0),(1,dim[1]))

		X = (X - numpy.matlib.repmat(mean,dim[0],1))/numpy.matlib.repmat(std,dim[0],1)
	elif flag == 3:
		minim = np.reshape(X.min(axis=0),(1,dim[1]))
		maxim = np.reshape(X.max(axis=0),(1,dim[1]))

		X = (X - numpy.matlib.repmat(minim,dim[0],1))/numpy.matlib.repmat((maxim-minim),dim[0],1)
	elif flag!=0:
		raise ValueError("%d is not a valid flag number"%flag)

	pcaTransf = np.dot(X, pcaComp[keep[0]:keep[1],:].T)
	# pcaComp[:keep[0],:] = 0
	# pcaComp[keep[1]:,:] = 0
	Xhat = np.dot(pcaTransf, pcaComp[keep[0]:keep[1],:])

	return Xhat

def allDataPca(x, keep = [0, 0], flag = 0):
	"""
	This function computes the transformed matrix x of 3D. This function is used to compute sources of channels and eliminate the
	desired ones over the all records.

	Input data:
		x - a 3D matrix [nr. records x nr. channels x nr]
		keep - 2 element list corresponding to the components desired to keep. From keep[0] to keep[1]
		flag - takes values 0, 1, 2 or 3. If flag = 0 the data will remain the same. If flag = 1, all observations of the
		data will be moved to 0 mean. If flag = 2, data will be standardized over features. If flag = 3, data will be
		normalized over features.

	Output data:
		xhat - the estimated x matrix after eliminating the desired components
	"""
	xhat = np.zeros((x.shape))
	comp = np.zeros((x.shape))
	for rec,i in zip(x, range(len(x))):
		# print("PCA: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(x)))
		comp[i,:,:] = pcaComp(rec)
		# if(np.corrcoef(comp[:2,:])[0,1]>0.97):
		xhat[i,:,:] = pcaTransform(rec, comp[i,:,:], keep = keep, flag = flag)
		# else:
		# xhat[i,:,:] = pcaTransform(rec, comp, keep = [0, len(x[0])], flag = flag)

	return xhat, comp

def heoallDataPca(x,heo, keep = [0, 0], flag = 0):
	"""
	This function computes the transformed matrix x of 3D. This function is used to compute sources of channels and eliminate the
	desired ones over the all records.

	Input data:
		x - a 3D matrix [nr. records x nr. channels x nr]
		keep - 2 element list corresponding to the components desired to keep. From keep[0] to keep[1]
		flag - takes values 0, 1, 2 or 3. If flag = 0 the data will remain the same. If flag = 1, all observations of the
		data will be moved to 0 mean. If flag = 2, data will be standardized over features. If flag = 3, data will be
		normalized over features.

	Output data:
		xhat - the estimated x matrix after eliminating the desired components
	"""
	xhat = np.zeros((x.shape))
	for rec,i in zip(x, range(len(x))):
		print("PCA: Wait... It might takes several minutes! You are at %d/%d"%(i+1,len(x)))
		comp = pcaComp(rec)
		for j in range(len(comp)):
			if(np.corrcoef(comp[j,:],heo[i,:])[0,1]>0.97):
				comp[j,:] = 0
				print("am intrat aici")

		pcaTransf = np.dot(rec, comp.T)
		xhat[i,:,:] = np.dot(pcaTransf, comp)

	return xhat

def allDataTransf(x, pcaComp, keep = [0, 0], flag = 0):
	xhat = np.zeros((x.shape))
	for rec,i in zip(x, range(len(x))):
		xhat[i,:,:] = pcaTransform(rec, pcaComp[i,:,:], keep = keep, flag = flag)

	return xhat
