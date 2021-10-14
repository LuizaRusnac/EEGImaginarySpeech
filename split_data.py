import numpy as np

"""
This split_data module contains:

split - splits data intro train and test with a rate of train_nr

split_kfold - splits the data set X into k folds

"""

def split(X, y, test_nr = 0.2, flag = 0, nr_cls = None, indexes = 0):
	"""
	This function splits data intro train and test with a rate of train_nr.

	Input data:
		X - EEG signals with dimension [nr. observations x nr. features] or [nr. observations x nr. channels x nr. features]
		y - target data. Dimension: [nr. observations]
			IMPORTANT: len(y) MUST be equal to len(X)!!!
		test_nr - the percentage of test data divided by 100. DEFAULT = 0.2 (20%)
		flag - is 0 or 1, if flag is 0 the split doesn't regard the class distribution, if flag is 1 the class will be split
			equally regarding the class target. DEFAULT = 0
		nr_cls - number of classes. If None, the classes targets will be computed using len(np.unique()). DEFAULT = None
		indexes - 0 or 1. If indexes is 0, the split indexes won't be returning. If indexes in 1, the function will return
			the indexes too. DEFAULT = 0

	Output data:
		xtrain - the training vectors
		ytrain - the training vectors target
		xtest - the test vectors
		ytest - the test vectors target

		optional:
			indxtrain - the indexes of the train vectors
			idxtest - the indexes of the test vectors

	"""

	if(len(X)!=len(y)):
		raise ValueError("X and y have different lengths")

	if(flag!= 0 and flag!=1):
		raise ValueError("It's not a valid flag number")

	if nr_cls is None:
		nr_cls = len(np.unique(y))

	dim = X.shape

	if flag==1:
		rnd_x = np.random.permutation(len(X))
		X = X[rnd_x]
		y = y[rnd_x]

		if len(dim)==3:
			xtrain = np.empty((0,dim[1],dim[2]))
			xtest = np.empty((0,dim[1],dim[2]))

		else:
			if len(dim)==2:
				xtrain = np.empty((0,dim[1]))
				xtest = np.empty((0,dim[1]))

			else:
				raise AttributeError("It's not a valid X matrix (X.shape is different from 2 or 3)")

		ytrain = np.empty((0))
		ytest = np.empty((0))

		if indexes==1:
			idxtrain = np.empty((0),dtype=int)
			idxtest = np.empty((0),dtype=int)

		for i in range(nr_cls):
		    X_interm = X[np.ravel(y==i)]
		    tr_word = len(X_interm) - int(len(X_interm)*test_nr)

		    if indexes==1:
		    	idx = rnd_x[np.ravel(y==i)]
		    	idxtrain = np.concatenate((idxtrain,idx[:tr_word]))
		    	idxtest = np.concatenate((idxtest,idx[tr_word:]))

		    xtrain = np.concatenate((xtrain,X_interm[:tr_word]))
		    xtest = np.concatenate((xtest,X_interm[tr_word:]))
		    ytrain = np.concatenate((ytrain,np.ones(len(X_interm[:tr_word]))*i))
		    ytest = np.concatenate((ytest,np.ones(len(X_interm[tr_word:]))*i))

	elif flag==0:
		rnd_x = np.random.permutation(len(X))
		X = X[rnd_x]
		y = y[rnd_x]

		nr_train = len(X) - int(len(X)*test_nr)

		if indexes==1:
			idxtrain = rnd_x[:tr_word]
			idxtest = rnd_x[tr_word:]

		xtrain = X[:nr_train]
		xtest = X[nr_train:]
		ytrain = y[:nr_train]
		ytest = y[nr_train:]

	if indexes==1:
		return xtrain, ytrain, xtest, ytest, idxtrain, idxtest
	else:
		return xtrain, ytrain, xtest, ytest

def split_kfold(X, y, k=5, flag = 0, nr_cls = None, indexes = 0):

	""" This function splits the data set X into k folds

	Input Data:
		X - is the input data with dimension NxM[xL] where N is the number of observations, M [and L] is the number of features
		y - is the target values of data X
		k - the number of folds to split the data, DEFAULT = 5
		flag - can have values of 0 or 1. When flag is 1, the algorithm splits the data conserving the class distribution, when flag is 0, 
		the data splits randomly without considering the number of observations in each class. DEFAULT = 0
		nr_cls - number of classes. If None, the classes targets will be computed using len(np.unique()) . DEFAULT = None
		indexes - 0 or 1. If indexes is 0, the split indexes won't be returning. If indexes in 1, the function will return
			the indexes too. DEFAULT = 0

	Output data:
		xtrain - the training vectors
		ytrain - the training vectors targe
		xtest - the test vectors
		ytest - the test vectors target

		optional:
			indxtrain - the indexes of the train vectors
			idxtest - the indexes of the test vectors
	"""
	if(len(X)!=len(y)):
		raise ValueError("X and y have different lengths")

	if(flag!= 0 and flag!=1):
		raise ValueError("It's not a valid flag number")

	if nr_cls is None:
		nr_cls = len(np.unique(y))

	if flag==1:

		X_train = []
		y_train = []
		X_test = []
		y_test = []

		if indexes==1:
			idxtrain = []
			idxtest = []


		for kval in range(k): # looping over the fold

			xtr = []
			ytr = []
			xtst = []
			ytst = []

			if indexes==1:
				idtr = []
				idtst = []

			for clas in range(nr_cls): # looping over the classes
				X_interm = X[np.ravel(y==clas)] # save all observations from class that equals clas in X_interm in order to split them 
												# equally further

				rnd = np.random.permutation(len(X_interm)) # make a random permutation of values, in case the observations are secvential
				X_interm = X_interm[rnd]

				nr_fold_cls = int(np.floor(len(X_interm)/k)) # computing the number of values in a folds for this class

				idx = kval*nr_fold_cls

				index_test = np.ravel(np.arange(idx,idx+nr_fold_cls)) # establish the the indexes from the X_interm which wil be for test
				index_train = np.arange(len(X_interm))
				index_train = np.ravel(np.delete(index_train, index_test)) # establish the the indexes from the X_interm which wil be for train

				xtr.extend(X_interm[index_train])					# concatenate the values every loop
				ytr.extend(np.ones((len(index_train),1))*clas)
				xtst.extend(X_interm[index_test])
				ytst.extend(np.ones((len(index_test),1))*clas)

				if indexes==1:
					idtr.extend(rnd[index_train])
					idtst.extend(rnd[index_test])

			X_train.append(xtr) # save every fold vectors
			y_train.append(ytr)
			X_test.append(xtst)
			y_test.append(ytst)

			if indexes==1:
				idxtrain.append(idtr)
				idxtest.append(idtst)

	if flag == 0:

		X_train = []
		y_train = []
		X_test = []
		y_test = []

		if indexes==1:
			idxtrain = []
			idxtest = []

		for kval in range(k):

			nr_fold_cls = int(np.floor(len(X)/k)) # computing number of observations per fold
			idx = kval*nr_fold_cls # computing the index of fold

			rnd = np.random.permutation(len(X)) # do a random permutation of the vectors in case there are secvential
			X = X[rnd]
			y = y[rnd]

			index_test = np.ravel(np.arange(idx,idx+nr_fold_cls)) # determine da indexes for the test
			index_train = np.arange(len(X)) 
			index_train = np.ravel(np.delete(index_train, index_test)) # determine the indexes for the train

			X_train.append(X[index_train]) # save observatons for every fold
			y_train.append(y[index_train])
			X_test.append(X[index_test])
			y_test.append(y[index_test])

			if indexes==1:
				idxtrain.append(rnd[index_train])
				idxtest.append(rnd[index_test])

	X_train = np.asarray(X_train) # transpose list into array
	X_test = np.asarray(X_test)
	y_train = np.ravel(np.asarray(y_train))
	y_test = np.ravel(np.asarray(y_test))

	if indexes==1:
		idxtrain = np.asarray(idxtrain)
		idxtest = np. asarray(idxtest)

	if indexes==1:
		return X_train, y_train, X_test, y_test, idxtrain, index_test
	else:
		return X_train, y_train, X_test, y_test