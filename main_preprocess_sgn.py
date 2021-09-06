import numpy as np
import pca
import lms
import preprocessing
import EEGPlot

xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\Xtrain.npy")
xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\Xtest.npy")
ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain.npy")
ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest.npy")

xtrain0, ytrain0 = preprocessing.spWin(xtrain, 1000, y=ytrain)
xtest0, ytest0 = preprocessing.spWin(xtest, 1000, y=ytest)

xtrain_pca = pca.allDataPca(xtrain0,keep=[2,len(xtrain[0])],flag=2)
xtest_pca = pca.allDataPca(xtest0,keep=[2,len(xtrain[0])],flag=2)

print(xtrain_pca)
print(xtest_pca)

np.save("Xtrain_pca.npy",xtrain_pca)
np.save("Xtest_pca.npy",xtest_pca)
np.save("ytrain_pca.npy",ytrain0)
np.save("ytest_pca.npy",ytest0)

# EEGPlot.eegGR(xtrain0[0], ch = [10], name = 'sgn_raw.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)
# EEGPlot.eegGR(xtrain_pca[0], ch = [10], name = 'sgn_pca.png', ch_labels = [0], path = None, flag = 0, space = 'maxim',flag2=0, fs = 1000)

# heo = np.load(r"C:\D\Doctorat\EEGSpeech\KaraOne_EEGSpeech_HEO_noLPF.npy")
# idxtrain = np.load("idxtrain.npy").astype(int)
# idxtest = np.load("idxtest.npy").astype(int)

# print(idxtrain)
# print(idxtest)

# heotrain = heo[idxtrain]
# heotest = heo[idxtest]

# y, xtrain_lms, w = lms.my_lms_data(xtrain, heotrain, L=400, mu=0.001, flag=1, repeat=15)
# y, xtest_lms, w = lms.my_lms_data(xtest, heotest, L=400, mu=0.001, flag=1, repeat=15)

# np.save("Xtrain_lms.npy",xtrain_lms)
# np.save("Xtest_lms.npy",xtest_lms)