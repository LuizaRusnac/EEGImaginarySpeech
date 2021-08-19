import numpy as np
import log
import deepLearning as dl
import matplotlib.pyplot as plt
import preprocessing
import datetime

method = 'PCA'
task = 'no task'
epochs = 2
LR = 0.0001
layers_nr = [64, 32, 32, 16]
mask = (2,2)
act = 'relu'
llact = 'softmax'
droput = [.1, .1]
R2 = 0
kernel_initializer='random_normal'
bias_initializer = 'random_normal'

if method == 'PCA':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain_pca.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xftest_pca.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\ytrain_pca.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\ytest_pca.npy")

if method == 'No Filter':
	xtrain = np.load(r"D:\TheodorRusnac\luiza_scripts\xftrain.npy")
	xtest = np.load(r"D:\TheodorRusnac\luiza_scripts\xftest.npy")
	ytrain = np.load(r"D:\TheodorRusnac\luiza_scripts\yftrain.npy")
	ytest = np.load(r"D:\TheodorRusnac\luiza_scripts\yftest.npy")

if task!= 'no task':
	ytrain = KOtasks.task(ytrain, task)
	ytest = KOtasks.task(ytest, task)

xtrain, minim, maxim = preprocessing.featureNorm(xtrain, flag=1)
xtest = preprocessing.featureNorm(xtest, minim, maxim)

xtrain,ytrain = dl.randperm(xtrain,ytrain)
xtest,ytest = dl.randperm(xtest,ytest)

nr_cls = len(np.unique(ytrain))

ytr = dl.createOutput(ytrain, nr_cls)
ytst = dl.createOutput(ytest, nr_cls)

dim = xtrain.shape

xtrain = xtrain.reshape((-1,dim[1],dim[2],1))
xtest = xtest.reshape((-1,dim[1],dim[2],1))

xshape = xtrain.shape

model = dl.CNN(xshape, nr_cls, LR)

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
	bias_initializer,epochs,history.history['accuracy'][-1], epochs,test_acc)

log.wlog("log_server.txt",text = text, flag = 1)

current_time = datetime.datetime.now() 
current_time = current_time.strftime("%d-%b-%Y_%H-%M-%S")
print(current_time)

name = "model_"+current_time

model.save(name)