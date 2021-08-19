import numpy as np
import split_data

x = np.load(r"C:\D\Doctorat\EEGSpeech\KaraOne_EEGSpeech_X_noLPF.npy")
y = np.load(r"C:\D\Doctorat\EEGSpeech\KaraOne_EEGSpeech_y_noLPF.npy")

xtrain, ytrain, xtest, ytest, idxtrain, idxtest = split_data.split(x, y, test_nr = 0.2, flag = 1, indexes = 1)

np.save("Xtrain.npy",xtrain)
np.save("ytrain.npy",ytrain)
np.save("Xtest.npy",xtest)
np.save("ytest.npy",ytest)
np.save("idxtrain.npy",idxtrain)
np.save("idxtest.npy",idxtest)

print(xtrain.shape)
print(xtest.shape)
print(idxtrain.shape)
print(idxtest.shape)