import numpy as np
import split_data

x = np.load(r"KaraOne_EEGSpeech_X_noLPF.npy")
y = np.load(r"KaraOne_EEGSpeech_y_noLPF.npy")
heo = np.load(r"KaraOne_EEGSpeech_HEO_noLPF.npy")

xtrain, ytrain, xtest, ytest, idxtrain, idxtest = split_data.split(x, y, test_nr = 0.2, flag = 1, indexes = 1)

print(idxtrain)
heotrain = heo[idxtrain]
heotest = heo[idxtest]

np.save("Xtrain_art3.npy",xtrain)
np.save("ytrain_art3.npy",ytrain)
np.save("Xtest_art3.npy",xtest)
np.save("ytest_art3.npy",ytest)
np.save("idxtrain_art3.npy",idxtrain)
np.save("idxtest_art3.npy",idxtest)
np.save("heotrain_art3.npy",heotrain)
np.save("heotest_art3.npy",heotest)

print(xtrain.shape)
print(xtest.shape)
print(idxtrain.shape)
print(idxtest.shape)
print(heotrain.shape)
print(heotest.shape)