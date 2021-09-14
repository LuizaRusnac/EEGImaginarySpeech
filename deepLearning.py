"""
This module contains:

createOutput - modifies the output to correspond to deep learning algorithms outputs

randperm - function rearange the vectors in dataset X

CNN - creates a CNN model

"""
import numpy as np
import tensorflow as tf, os
from tensorflow.keras import datasets, layers, models

def createOutput(y, nr_clases):
	"""
	This function modifies the output to correspond to deep learning algorithms outputs

	Data Input:
		y - the target vector. Dimension: [nr. observations x 1]
		nr_clases - number of different classes in dataset

	Data Output:
		yout - the modified y vector into matrix. Dimension: [nr. observations x nr. clases]
	"""
	yout = np.zeros((len(y),nr_clases))
	for j in range(len(y)):
	        yout[j, int(y[j]) ] = 1

	return yout

def randperm(X, y):
	"""
	This function rearange the vectors in dataset X

	Data Input:
		X - the dataset. Dimension: [nr. observations x nr. fatures]
		y - the target vector. Dimension: nr. observations

	Data Output:
		Xrnd - the dataset with rearanged vectors
		yrnd - the rearanged target vectors regarding X dataset
	"""
	rnd = np.random.permutation(int(len(X)))
	Xrnd = X[rnd]
	yrnd = y[rnd]

	return Xrnd, yrnd


def CNN(xshape, cls_nr, LR, layers_nr = [64, 32, 32, 16], mask = (2,2), act = 'relu', llact = 'softmax', droput = [.5, .5], R2 = 0, kernel_initializer='random_normal', bias_initializer = 'random_normal'):
	"""
	This function creates a CNN model

	Input Data
		xshape - the shape of the input data. Dimesion: [4 x 1]
		cls_nr - number of classes to set the number of neurons in output layer
		LR - learning rate
		layers_nr - number of neurons in each layer. Dimesion: [3 x 1]. DEFAULT =  [64, 32, 32, 16]
		mask - the dimesion of maskfor CNN layers. DimesionL [1 x 2]. DEFAULT = (2,2)
		act - activation function. DEFAULT = 'relu'
		llact - last layer activation function. DEFAULT = 'softmax'
		dropout - the droput for the layers. DEFAULT = [.5, .5]
		R2 - regularization value. DEFAULT = 0
		kernel_initializer - how the kernel it's initialized. DEFAULT = 'random_normal' 
		bias_initialier - DEFAULT = 'random_normal'

	Output Data:
		model - returns the created model
	"""

	model = models.Sequential()
	model.add(layers.Conv2D(layers_nr[0], mask, activation = act, input_shape=(xshape[1], xshape[2], xshape[3]), kernel_regularizer=tf.keras.regularizers.l2(R2), kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Conv2D(layers_nr[1], mask, activation = act, kernel_regularizer=tf.keras.regularizers.l2(R2), kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Flatten())
	model.add(layers.Dense(layers_nr[2], activation = act,kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))
	model.add(layers.Dropout(droput[0]))
	model.add(layers.Dense(layers_nr[3], activation = act,kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))
	model.add(layers.Dropout(droput[1]))
	if (llact == 'none'):
	    model.add(layers.Dense(cls_nr, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))
	else:
	    model.add(layers.Dense(cls_nr, activation = llact,kernel_initializer = kernel_initializer, bias_initializer = bias_initializer))

	model.compile(tf.optimizers.Adam(LR),loss = tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])

	return model

