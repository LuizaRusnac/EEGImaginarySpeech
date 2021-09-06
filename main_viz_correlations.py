import numpy as np
from EEGPlot import eegGR
from scipy import signal
import matplotlib.pyplot as plt
from preprocessing import sgnNorm, sgnStd

x = np.load("Xtrain.npy")
y = np.load("ytrain.npy")

iy_signals = x[y==0]
pot_signals = x[y==8]

db_coef = signal.daub(2)
val, phi, db = signal.cascade(db_coef, 5)

db = np.reshape(db,(1,db.shape[0]))


iy = np.reshape(iy_signals[0,60,:],(1,4000))
iy = sgnNorm(iy)
pot = np.reshape(pot_signals[0,60,:],(1,4000))
pot = sgnNorm(pot)

corr_iy = signal.convolve(iy,db,'same')
corr_pot = signal.convolve(pot,db,'same')

# print(corr_pot.shape)

fsgn = np.concatenate((iy,pot,corr_iy, corr_pot),axis=0)

# iy = iy_signals[0,60,:]
# iy = sgnNorm(iy)
# db = np.resize(db,(1,4000))
# db = sgnStd(db)

print(iy.shape)
print(db.shape)

eegGR(fsgn, ch=[4], name = "signals_raw_and_corr_db4.png",space = 'maxim')

plt.figure()
plt.plot(db.T)
plt.show()
plt.close()


plt.figure()
plt.plot(iy[:500].T)
plt.show()
plt.close()

plt.figure()
corr_iy_full = signal.convolve(iy.T,db.T,'full')
plt.plot(corr_iy_full.T)
plt.show()
plt.close()

print(corr_iy.max())
print(corr_pot.max())
print(corr_iy_full.max())