import numpy as np
import matplotlib.pyplot as plt
import preprocessing


def save_plot(examples, n, m, name):
	# plot images
	for i in range(n * m):
		# define subplot
		plt.subplot(n, m, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :])
	plt.savefig(name)
	plt.show()

def choose_vect(x, y, nr_cls, n):
	xfin = np.zeros((nr_cls*n,x.shape[1],x.shape[2]))
	for i in range(nr_cls):
		xinterm = x[np.ravel(y==i),:,:]
		xfin[(i*n):(i*n)+n,:,:] = xinterm[:n,:,:]

	return xfin

X = np.load(r"D:\TheodorRusnac\luiza_scripts\xtrain_pca.npy")
y = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")

xfpca = choose_vect(X,y,5,5)
save_plot(xfpca,5,5,"img_pca_features.png")

X = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")
y = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")

xpca = choose_vect(X,y,5,5)
save_plot(xpca,5,5,"img_pca_sgn.png")


X = np.load(r"D:\TheodorRusnac\luiza_scripts\xtrain.npy")
y = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain.npy")

x = choose_vect(X,y,5,5)
save_plot(xfpca,5,5,"img_features.png")

X = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain.npy")
y = np.load(r"D:\TheodorRusnac\luiza_scripts\yftrain.npy")

x = choose_vect(X,y,5,5)
save_plot(xfpca,5,5,"img_sgn.png")
