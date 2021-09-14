from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import matplotlib.pyplot as plt
import preprocessing
import numpy as np
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=11):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# create and save a plot of generated images
def save_plot(examples, n, name):
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :, 0])
	plt.show()
	plt.save(name)

def load_real_samples():
	# load dataset
	X = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")
	trainy = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")
	# X = trainX[np.ravel(trainy==0)]
	# expand to 3d, e.g. add channels
	X = np.resize(X,(X.shape[0],28,28))
	# convert from ints to floats
	# X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = preprocessing.featureNorm(X)
	X = X.reshape((-1,X.shape[1],X.shape[2],1))

	return [X, trainy]

 
# load model
model = load_model('cgan_generator.h5')
# generate images
[latent_points, labels] = generate_latent_points(100, 100)
# specify labels
# labels = asarray([x for _ in range(10) for x in range(10)])
# generate images
print(latent_points.shape)
# print(labels.shape)
X  = model.predict([latent_points, labels])
print(X.shape)
real = load_real_samples()
# scale from [-1,1] to [0,1]
# X = (X + 1) / 2.0
# plot the result
save_plot(X, 11, 'fake_img.png')
save_plor(real,11,'real_img.png')

