import numpy as np
import pca
import lms
import preprocessing

xtrain = np.load("Xtrain.npy")
xtest = np.load("Xtest.npy")
ytrain = np.load("ytrain.npy")
ytest = np.load("ytest.npy")

# xtrain0, ytrain0 = preprocessing.spWin(xtrain, 1000, y=ytrain)
# xtest0, ytest0 = preprocessing.spWin(xtest, 1000, y=ytest)

# xtrain_pca = pca.allDataPca(xtrain0,keep=[2,len(xtrain[0])])
# xtest_pca = pca.allDataPca(xtest0,keep=[2,len(xtrain[0])])

# np.save("Xtrain_pca.npy",xtrain_pca)
# np.save("Xtest_pca.npy",xtest_pca)
# np.save("ytrain_pca.npy",ytrain0)
# np.save("ytest_pca.npy",ytest0)

heo = np.load(r"C:\D\Doctorat\EEGSpeech\KaraOne_EEGSpeech_HEO_noLPF.npy")
idxtrain = np.load("idxtrain.npy").astype(int)
idxtest = np.load("idxtest.npy").astype(int)

print(idxtrain)
print(idxtest)

heotrain = heo[idxtrain]
heotest = heo[idxtest]

y, xtrain_lms, w = lms.my_lms_data(xtrain, heotrain, L=400, mu=0.001, flag=1, repeat=15)
y, xtest_lms, w = lms.my_lms_data(xtest, heotest, L=400, mu=0.001, flag=1, repeat=15)

np.save("Xtrain_lms.npy",xtrain_lms)
np.save("Xtest_lms.npy",xtest_lms)