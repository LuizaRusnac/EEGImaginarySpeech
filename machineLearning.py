from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np

"""
This module contains:

	knn - 
"""
def knn(xtrain,ytrain,xtest,ytest, kval = [1], flag = 0):
	"""
	This function compute knn for different values of k using sklearn

	Input Data:
		xtrain - training set. Dimension: [nr. observations x nr. features]
		ytrain - training target. Dimension: [nr. observations x 1]
		xtest - testing set. Dimension: [nr. observations x nr. features]
		ytest - test target. Dimension: [nr. observations x 1]
		kval - desired values of k. If len(kval)=1 the function will return the sore for the k value. If len(kval)=2 will be
			computed the scores from k[0] to k[1]. If len(kval)=3 will be computed the scores between k[0] and k[1] with
			step k[2]. DEFAULT = [1]
		flag - takes values of 0 or 1. If flag is 0, the scors for every k won't be plotted. If flag is 1, will be plotted
			the scores for k. DEFAULT = 0
			IMPORTANT: The flag CAN BE 1 ONLY if len(kval)>1!

	Output Data:
		k - the scores for desired kvals

	"""
	k = []

	if len(kval)==1:
		clf = neighbors.KNeighborsClassifier(kval[0], weights='distance')
		clf.fit(xtrain, ytrain)
		score = clf.score(xtest,ytest)
		print("---Accuracy: %s ---" %score)
		return score

	else:
		if len(kval)==2:
			for i in range(kval[0],kval[1],2):
			    clf = neighbors.KNeighborsClassifier(i, weights='distance')
			    clf.fit(xtrain, ytrain)
			    score = clf.score(xtest,ytest)
			    k = np.append(k, score)
			    print("---Neighbour: %s --- Accuracy: %s ---" %(i, score))

		else:
			if len(kval)==3:
				for i in range(kval[0],kval[1],kval[2]):
				    clf = neighbors.KNeighborsClassifier(i, weights='distance')
				    clf.fit(xtrain, ytrain)
				    score = clf.score(xtest,ytest)
				    k = np.append(k, score)
				    print("---Neighbour: %s --- Accuracy: %s ---" %(i, score))

			else:
				raise ValueError("Too many values for kval")

	if len(kval)==1 and flag==1:
		AttributeError("Can't be plotted one value!")

	if(flag==1):
		if len(kval)==2:
			plt.figure()
			plt.plot(k)
			plt.xlabel("K values")
			plt.ylabel("Scores")
			plt.savefig("scores_for_kvalues.png")
			plt.close()
		if len(kval)==3:
			xval = np.ravel(np.arange(kval[0],kval[1],kval[2]))
			plt.figure()
			plt.plot(xval,k)
			plt.xlabel("K values")
			plt.ylabel("Scores")
			plt.savefig("scores_for_kvalues.png")
			plt.close()

	return k
