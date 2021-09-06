import numpy as np
import matplotlib.pyplot as plt
import preprocessing

"""
This module contains:

eegGR - plot multiple channels of EEG data

"""

def eegGR(X, ch = None, name = 'eegGR.png', ch_labels = [0], path = None, flag = 0, space = 'minim',flag2=0, fs = 1000):
	"""
	This function is designed to plot multiple channels of EEG data

	Input data:
		X - The EEG data. Dimension: [nr. channels x nr. samples]
		ch - the channels to be ploted. If ch is None, all channels will be ploted. If len(ch)=1, channels from 0 to ch will be
			ploted. If len(ch)=2 channels from ch[0] to ch[1] will be plotted. DEFAULT = None
		name - the name of the file that will be saved with the plot. DEFAULT = 'eegGR.png'
		ch_labels - the labels names of the channels. DEFAULT = [0]
		path - the path were the image will be saved. DEFAULT = None
		flag - it takes values 0 or 1. If flag=0 the data will be standardized. If flag=1 the data will be normalized. DEFAULT = 0
		space - the space between the channel reprezentation. It can take values: 'minim', 'maxim', 'var'. If space is 'minim',
			the space will be computed as X.max(axis=1).min(). If space is 'maxim', the space will be computed as 
			maxim = X.max(axis=1).max(). If space is 'var', the space will be computed as X.min(axis=1).min(). DEFAULT = 'minim'
		flag2 - takes values of 0 or 1. If flag2 is 0, the signal will have samples on xlabel. If flag2 = 1, the signal will 
			have time on xlabel. DEFAULT = 0.
		fs - the frequency samples of the signal to compute time vector. DEFAULT = 1000.

		IMPORTANT: The length og ch_labels MUST be the same as X.
				The fs is needed only if flag2 is 1.

	"""
	if(flag2==1):
		time = np.arange(len(X[0]))/fs
	elif flag2==0:
		time = np.arange(len(X[0]))

	plt.figure()

	if flag==0:
		X = preprocessing.sgnStd(X)
	else:
		if flag==1:
			X = preprocessing.sgnNorm(X)
		else:
			if flag>2:
				raise ValueError("Flag is not a valid value")

	if space == 'minim':
		maxim = X.max(axis=1).min()
	else:
		if space == 'maxim':
			maxim = X.max(axis=1).max()
		else:
			if space=='var':
				maxim = X.min(axis=1).min()
			else:
				raise ValueError("Space is not a valid value")

	if ch is None:
		for i, reverse in zip(range(len(X)),range(len(X)-1,0,-1)):
			plt.plot(X[i,:]+(maxim*reverse))
			if ch_labels[0] != 0:
				plt.text(-1,maxim*reverse,ch_labels[i])
	else:	
		if len(ch)==1:
			if ch[0]<0:
				for i, reverse in zip(range(len(X)+ch[0]),range(len(X)+ch[0],0,-1)):
					plt.plot(X[i,:]+(maxim*reverse))
					if ch_labels[0] != 0:
						if (len(X)!=len(ch_label)):
							raise ValueError("Different lengtsh for X and ch_label")
						plt.text(-1,(maxim*reverse),ch_label[i])
			else:
				for i, reverse in zip(range(ch[0]),range(ch[0],0,-1)):
					plt.plot(time,X[i,:]+(maxim*reverse))
					if ch_labels[0] != 0:
						if (len(X)!=len(ch_labels)):
							raise ValueError("Different lengtsh for X and ch_label")
						plt.text(0,(maxim*reverse),ch_labels[i])
		else:
			if len(ch)==2:
				for i, reverse in zip(range(ch[0]-1,ch[1]-1),range(ch[1]-1,ch[0]-1,-1)):
						plt.plot(time,X[i,:]+(maxim*reverse))
						if ch_labels[0] != 0:
							if (len(X)!=len(ch_labels)):
								raise ValueError("Different lengths for X and ch_label")
							plt.text(0,(maxim*reverse),ch_labels[i])

			else:
				raise ValueError("ch must be a list with maxim two values")

	if flag2==0:
		plt.xlabel("Nr. samples")
		plt.ylabel("Amplitude")
	if flag2==1:
		plt.xlabel("Time [s]")
		plt.ylabel("Amplitude")
		
	if path is None:
		plt.savefig(name)
	else:
		final_path = path + '/' + name
		plt.savefig(final_path)

	plt.show()
	plt.close()



