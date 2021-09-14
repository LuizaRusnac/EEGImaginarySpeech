import numpy as np
from EEGPlot import eegGR
from scipy import signal
import matplotlib.pyplot as plt
from preprocessing import sgnNorm, sgnStd, featureStd

x = np.load("Xtrain.npy")
xt = np.load("Xtest.npy")

db_coef = signal.daub(4)
val, phi, db = signal.cascade(db_coef, 8)
db = np.reshape(db,(db.shape[0],1))

# t = np.linspace(-1, 1, 500, endpoint=False)
# db = np.reshape(signal.gaussian(500,3),(500,1))
db = featureStd(db)*(1/np.sqrt(2*3.14))
print(db.shape)

# db = signal.ricker(500, 8)

plt.figure()
plt.plot(db)
plt.show()
plt.close()

xfdbtrain = np.zeros(x.shape)
for i in range(len(x)):
	sgn = sgnStd(x[i,:,:])
	for j in range(len(x[0])):
		xfdbtrain[i,j,:] = signal.convolve(np.reshape(sgn[j,:],(1,sgn.shape[1])),db.T,'same')

plt.figure()
plt.plot(xfdbtrain[0,35,:])
plt.show()
plt.close()

xfdbtest = np.zeros(xt.shape)
for i in range(len(xt)):
	sgn = sgnStd(xt[i,:,:])
	for j in range(len(xt[0])):
		xfdbtest[i,j,:] = signal.convolve(np.reshape(sgn[j,:],(1,sgn.shape[1])),db.T,'same')

np.save("xfdbtrain",xfdbtrain)
np.save("xfdbtest",xfdbtest)
