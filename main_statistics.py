import numpy as np
import pandas as pd
import statistics


ch = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
index =['/iy/', '/uw/','/piy/','/tiy/','/diy/','/m/','/n/','pat','pot', 'knew', 'gnaw']
columns = []

for i in range(len(ch)):
	string = ch[i] + '_minim'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_maxim'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_stdev'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_entropy'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_power'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_max_corr_ricker'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_max_corr_db2'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_max_corr_db4'
	columns.append(string)

for i in range(len(ch)):
	string = ch[i] + '_mean_of_max_corr_db6'
	columns.append(string)

print(len(columns))

xtrain = np.load(r"Xtrain.npy")
ytrain = np.load(r"ytrain.npy")

dataf = np.zeros((len(index),len(columns)))

for cls in range(11):

	x = xtrain[ytrain==cls]

	minim = statistics.minim(x)
	maxim = statistics.maxim(x)
	mean = statistics.mean(x)
	stdev = statistics.stdev(x)

	entropy = np.zeros((x.shape[:2]))
	max_corr_ricker = np.zeros((x.shape[:2]))
	max_corr_db2 = np.zeros((x.shape[:2]))
	max_corr_db4 = np.zeros((x.shape[:2]))
	max_corr_db6 = np.zeros((x.shape[:2]))

	for i in range(len(x)):
		for j in range(len(x[0])):
			entropy[i,j] = statistics.entropy(x[i,j,:])
			max_corr_ricker[i,j] = statistics.max_corr_ricker(x[i,j,:], points=300, a=4)
			max_corr_db2[i,j] = statistics.max_corr_db(x[i,j,:], order = 2)
			max_corr_db4[i,j] = statistics.max_corr_db(x[i,j,:], order = 4)
			max_corr_db6[i,j] = statistics.max_corr_db(x[i,j,:], order = 6)

	mean_of_entropy = statistics.mean_of_entropy(entropy)
	power = statistics.power(x)
	mean_of_power = statistics.mean_of_power(power)
	mean_of_max_corr_ricker = statistics.mean_of_max_corr_ricker(max_corr_ricker)
	mean_of_max_corr_db2 = statistics.mean_of_max_corr_ricker(max_corr_db2)
	mean_of_max_corr_db4 = statistics.mean_of_max_corr_ricker(max_corr_db4)
	mean_of_max_corr_db6 = statistics.mean_of_max_corr_ricker(max_corr_db6)

	data = np.concatenate((minim, maxim, mean, stdev, mean_of_entropy, mean_of_power, mean_of_max_corr_ricker, mean_of_max_corr_db2, mean_of_max_corr_db4, mean_of_max_corr_db6), axis=1)
	dataf[cls,:] = data

df = pd.DataFrame(dataf, columns = columns, index = index)
pd.options.display.max_columns = None

df.to_excel("KOstatistics")
print(df.iloc[:,-62*4:-62*2])
