import numpy as np
from scipy import signal
from math import log, e

def minim(x):
	return np.reshape(x.min(axis=0).min(axis=1),(1,x.shape[1]))

def maxim(x):
	return np.reshape(x.max(axis=0).max(axis=1),(1,x.shape[1]))

def mean(x):
	return np.reshape(x.mean(axis=0).mean(axis=1),(1,x.shape[1]))

def stdev(x):
	return np.reshape(x.std(axis=0).std(axis=1),(1,x.shape[1]))

def entropy(x, base = None):
	value,counts = np.unique(x, return_counts=True)
	norm_counts = counts / counts.sum()
	base = e if base is None else base
	return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def mean_of_entropy(x):
	return np.reshape(x.mean(axis=0),(1,x.shape[1]))

def power(x):
	return (x**2).sum(axis=2)/x.shape[2]

def mean_of_power(x):
	return np.reshape(x.mean(axis=0),(1,x.shape[1]))

def max_corr_ricker(x, points=100, a=4):
	 ricker = signal.ricker(points, a)
	 corr = signal.correlate(x,ricker)
	 return corr.max()

def mean_of_max_corr_ricker(x):
	return np.reshape(x.mean(axis=0),(1,x.shape[1]))

def max_corr_db(x, order = 2):
	 db_coef = signal.daub(order)
	 val, phi, db = signal.cascade(db_coef, 10)
	 corr = signal.correlate(x,db)
	 return corr.max()