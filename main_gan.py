from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import numpy as np
import preprocessing
import matplotlib.pyplot as plt
 
# define the standalone discriminator model
def define_discriminator(in_shape=(62,62,1)):
	model = Sequential()
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	# model.add(Dense(6272, activation = 'relu'))
	model.summary()
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 31 * 31
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((31, 31, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (2,2), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (2,2), strides=(1,1), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# generate
	model.add(Conv2D(1, (31,31), activation='tanh', padding='same'))
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
	return model
 
# load fashion mnist images
def load_real_samples():
	# load dataset
	# trainX = np.load(r"C:\D\Doctorat\EEGImaginarySpeech\Xtrain_pca.npy")
	# trainy = np.load(r"C:\D\Doctorat\EEGImaginarySpeech\ytrain_pca.npy")
	# trainX = trainX[np.ravel(trainy==0)]
	trainX = np.load(r"xftrain_pca_art3.npy")
	trainy = np.load(r"xftrain_pca_art3.npy")
	# X = trainX[np.ravel(trainy==0)]
	# expand to 3d, e.g. add channels
	# X = np.resize(X,(X.shape[0],28,28))
	# convert from ints to floats
	# X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	# X = preprocessing.featureNorm(X)
	# X = trainX
	X = np.zeros((trainX.shape[0],trainX.shape[1],trainX.shape[1]))
	for i in range(len(trainX)):
		X[i,:,:] = np.cov(trainX[i,:,:]);
	print("************Dim Xtrain = **************")
	print(trainX.shape)
	X = X.reshape((-1,X.shape[1],X.shape[2],1))

	return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n_samples, 1))
	return X, y
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1, n_batch=128):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, acc1 = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, acc2 = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# print("************** Print X latent space ************")
			# print(X_gan)
			# print("**************")
			# print("************** Print X latent space dim************")
			# print(X_gan.shape)
			# print("**************")
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss, accg = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f ---- acc1=%.4f, acc2=%.4f, accg=%.4f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss,acc1,acc2,accg))
	# save the generator model
	g_model.save('generator.h5')

def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0])
	pyplot.show()
 
	gan_model.save('generator.h5')

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

 
# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)

# load model
# model = load_model('cgan_generator.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# specify labels
labels = np.asarray([x for _ in range(2) for x in range(50)])
# generate images
X  = gan_model.predict([latent_points, labels])

save_plot(X, 10, 'fake_image.png')
save_plot(dataset,10, 'real_image.png')