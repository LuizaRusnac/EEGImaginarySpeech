from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
import numpy as np
import preprocessing
import matplotlib.pyplot as plt

# define the standalone discriminator model
def define_discriminator(in_shape=(62,62,1), n_classes=11):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 500)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim, n_classes=11):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 500)(in_label)
	# linear multiplication
	n_nodes = 31 * 31
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((31, 31, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 31 * 31
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((31, 31, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(1,1), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (31,31), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
	return model
 
# load fashion mnist images
def load_real_samples():
	# load dataset
	X = np.load(r"xtrain_1000.npy")
	trainy = np.load(r"ytrain_1000.npy")
	# X = trainX[np.ravel(trainy==0)]
	# expand to 3d, e.g. add channels
	# X = np.resize(X,(X.shape[0],60,500))
	xc = np.zeros((X.shape[0],X.shape[1],X.shape[1]))
	for i in range(len(X)):
		xc[i,:,:] = np.cov(X[i,:,:])
	# convert from ints to floats
	# X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = preprocessing.featureStd(xc)
	X = X.reshape((-1,X.shape[1],X.shape[2],1))

	return [X, trainy]

 # load fashion mnist images
# def load_test_samples():
# 	# load dataset
# 	X = np.load(r"xftest_pca.npy")
# 	trainy = np.load(r"ytest_pca.npy")
# 	# X = trainX[np.ravel(trainy==0)]
# 	# expand to 3d, e.g. add channels
# 	# X = np.resize(X,(X.shape[0],60,500))
# 	# convert from ints to floats
# 	# X = X.astype('float32')
# 	# scale from [0,255] to [-1,1]
# 	X = preprocessing.featureStd(X)
# 	X = X.reshape((-1,X.shape[1],X.shape[2],1))

	# return [X, trainy]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=11):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# print("************")
			# print(z_input.shape)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss, acc = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f ----- acc: %.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, acc))
	# save the generator model
	gan_model.save('cgan_generator.h5')
	g_model.save('generator_model.h5')


def save_plot(examples, n, name):
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :, 0])
	plt.savefig(name)
	plt.show()
	
 
# size of the latent space
latent_dim = 1000
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
gan_model.summary()
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

# testset = load_test_samples()

[latent_points, labels] = generate_latent_points(1000, 110)
print(labels)
labels = np.asarray([x for _ in range(10) for x in range(11)])
print(labels.shape)
X  = gan_model.predict([latent_points, labels])
print(X)
print(labels)
# intermediate_layer_model = gan_model(inputs=gan_model.input,
#                                  outputs=gan_model.get_layer('conv2d_2').output)
# intermediate_output = intermediate_layer_model.predict([latent_points, labels])

# for i in gan_model.layers:
# 	print(i.output)

# intermediate_layer_model = Model(inputs=gan_model.layers[:2],
#                                  outputs=[l.output for l in gan_model.layers[2:]])
# intermediate_output = intermediate_layer_model.predict([latent_points, labels])
# print(intermediate_output[12])

pred_img = g_model.predict([latent_points, labels])
print(pred_img)
print(dataset[0])

save_generated = np.squeeze(pred_img)
print(save_generated.shape)

np.save("generated_pca",save_generated)
np.save("generate_pca_labels",labels)

save_plot(pred_img, 3, 'fake_image.png')
save_plot(dataset[0],3, 'real_image.png')
# print(intermediate_output)
# acct = gan_model.evaluate(testset)
# print("***** ACC: %.4f *****"%acct)
# pred = gan_model.predict(testset)
# print("***** Predictions: *****")
# print(pred)
# print("**********")
# print("***** Real: *****")
# print(testset[1])
# print("**********")



