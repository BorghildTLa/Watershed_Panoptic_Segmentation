import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

LOG_FILE1 = '/hdd/bg/HAIL_angletransform/panoptic_2/MODELS/8/HR/log_8_HR.txt'
#LOG_FILE2 = '/hdd/bgbl/HAIL_angletransform/panoptic/MODELS/11/HR/log_11_HR.txt'

def get_log(log):
	f = open(log, 'r')
	lines = f.readlines()
	f.close()

	loss_o = []
	loss_a = []
	loss_sem = []
	loss_c = []
	for line in lines:
		loss_o.append(float(line.strip('\n').split(' ')[1].split(',')[0]))
		loss_a.append(float(line.strip('\n').split(' ')[2].split(',')[0]))
		loss_sem.append(float(line.strip('\n').split(' ')[3].split(',')[0]))
		loss_c.append(float(line.strip('\n').split(' ')[4].split(',')[0]))

	return loss_o,loss_a,loss_sem,loss_c

def plot_iteration(log1):
	loss_o,loss_a,loss_s,loss_c = get_log(log1)
	#loss2 = get_log(log2)
	loss_o = gaussian_filter1d(loss_o, sigma=100)
	loss_a = gaussian_filter1d(loss_a, sigma=100)
	loss_s = gaussian_filter1d(loss_s, sigma=100)
	loss_c = gaussian_filter1d(loss_c, sigma=100)

	#loss2 = gaussian_filter1d(loss2, sigma=150)
	#plt.subplot(221)
	plt.plot(range(len(loss_o)), loss_o)
	#plt.subplot(222)
	plt.plot(range(len(loss_a)), loss_a)
	#plt.subplot(223),
	plt.plot(range(len(loss_s)), loss_s)
	#plt.subplot(224)
	plt.plot(range(len(loss_c)), loss_c)
	#plt.plot(range(len(loss2)), loss2)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Training Curve')
	plt.show()

def plot_epoch(log, num_samples, batch_size):

	loss = get_log(log)[2]
	print(len(loss))
	epochs = len(loss) * batch_size // num_samples
	iters_per_epochs = num_samples // batch_size
	x = range(0, epochs+1)
	y = [loss[0]]
	for i in range(epochs):
		y.append(np.mean(np.array(loss[i*iters_per_epochs+1: (i+1)*iters_per_epochs+1])))
	plt.plot(x, y)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Curve')
	plt.show()

if __name__ == '__main__':
	#plot_epoch(LOG_FILE1,100480,4)
	plot_iteration(LOG_FILE1)
