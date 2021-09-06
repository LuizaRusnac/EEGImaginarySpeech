import numpy as np
import log
import deepLearning as dl
import matplotlib.pyplot as plt
import preprocessing
import datetime
import tensorflow as tf

method = 'No Filter'
task = 'no task'
epochs = 50
last_epochs = 0
LR = 0.00001
layers_nr = [128, 64, 64, 128]
mask = (2,2)
act = 'relu'
llact = 'softmax'
droput = [.2, .2]
R2 = 0
kernel_initializer='random_normal'
bias_initializer = 'random_normal'

model_path = None


if method == 'PCA':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xftest_pca.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest_pca.npy")

if method == 'No Filter':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\Xtrain.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\Xtest.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest.npy")

if task!= 'no task':
	ytrain = KOtasks.task(ytrain, task)
	ytest = KOtasks.task(ytest, task)

xtrain, minim, maxim = preprocessing.featureNorm(xtrain, flag=1)
xtest = preprocessing.featureNorm(xtest, minim, maxim)

print("***Xtrain:***")
print(xtrain)

print("***Xtest:***")
print(xtest)

xtrain,ytrain = dl.randperm(xtrain,ytrain)
xtest,ytest = dl.randperm(xtest,ytest)

nr_cls = len(np.unique(ytrain))

ytr = dl.createOutput(ytrain, nr_cls)
ytst = dl.createOutput(ytest, nr_cls)

dim = xtrain.shape

xtrain = xtrain.reshape((-1,dim[1],dim[2],1))
xtest = xtest.reshape((-1,dim[1],dim[2],1))

xshape = xtrain.shape

if model_path is None:
	model = dl.CNN(xshape, nr_cls, LR)
else:
	model = tf.keras.models.load_model(model_path)

history = model.fit(xtrain, ytr, epochs=epochs, validation_data=(xtest, ytst))

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("grapchic_loss.png")
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("grapchic_accuracy.png")
plt.close()

test_loss, test_acc = model.evaluate(xtest, ytst)
train_loss, train_acc = model.evaluate(xtrain, ytrain)

text = "%s 1s standardized signal before feature extraction \n\
Feature extraction: Spectrum all frequencies \n\
With CNN, Task: %s \n\
LR = %.5f, layers nr = %s, mask = %s, Act functions in layers: %s \n\
Act function in last layer: %s, Droput: %s, Regularization: %.4f \n\
Kernel initializer: %s, Bias initilizer: %s \n\
Optimizer: Adam, Loss: MeanSquaredError \n\
Train acc after %d iters: %.4f \n\
Val acc after %d iters: %.4f"%(method,task,LR,layers_nr, mask, act, llact, droput, R2, kernel_initializer, \
	bias_initializer,epochs+last_epochs,history.history['accuracy'][-1], epochs+last_epochs,test_acc)

log.wlog("log_server_cnn.txt",text = text, flag = 1)

current_time = datetime.datetime.now() 
current_time = current_time.strftime("%d-%b-%Y_%H-%M-%S")
print(current_time)

name = "model_"+current_time

model.save(name)