from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image

from network import *
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from matplotlib import pyplot as plt
from skimage.io import imsave

"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	# train
	def train(self):
		normal_color = "\033[0;37;40m"
		self.train_setup()

		self.sess.run(tf.global_variables_initializer())


		# Load the pre-trained model if provided
		if self.conf.pretrain_file is not None:
			self.load(self.loader, self.conf.pretrain_file)

		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# Train!
		for step in range(self.conf.num_steps+1):
			start_time = time.time()
			feed_dict = { self.curr_step : step }
			if step % self.conf.save_interval == 0:
				loss_value,lossa,loss_sem,loss_c, images, labels, preds, summary, _ = self.sess.run(
					[self.reduced_loss,self.ins_a_loss,self.sem_loss,self.ins_c_loss,
					self.image_batch,
					self.label_batch,
					self.pred,
					self.total_summary,
					self.train_op],
					feed_dict=feed_dict)



				self.summary_writer.add_summary(summary, step)
				self.save(self.saver, step)
			elif step % 500 == 0:
				loss_value,lossa,loss_sem,loss_c, images_s, labels,label_angles, preds,preds2,preds3,c1,c2, _ = self.sess.run(
					[self.reduced_loss,self.ins_a_loss,self.sem_loss,self.ins_c_loss,
					self.images_summary,self.labels_summary,self.label_angles,
					self.preds_summary,self.pred_2,self.pred_3,self.c1,self.c2,
					self.train_op],
					feed_dict=feed_dict)

				'''
				plt.subplot(121),plt.imshow(label_angles[0,:,:,0])
				plt.subplot(122),plt.imshow(label_angles[0,:,:,1])
				plt.show()
				'''

				for i in range(0,self.conf.batch_size):
					if np.sum(labels[i,:,:,:]) != 0:
						im = Image.fromarray(images_s[i,:,:,:],mode='RGB')
						im.save('prediction/' +str(step)+ '_image_'+str(i)+'.png')

						im = Image.fromarray(labels[i,:,:,:], mode='RGB')
						im.save('prediction/' +str(step)+ '_label_'+str(i)+'.png')

						im = Image.fromarray(preds[i,:,:,:], mode='RGB')
						im.save('prediction/' +str(step)+ '_p_sem_'+str(i)+'.png')

						filler=np.zeros(np.shape(preds2[i,:,:,0]))
						preds2[i,:,:,0]=(preds2[i,:,:,0]/np.max(preds2[i,:,:,0]))*255
						preds3[i,:,:,0]=(np.clip(preds3[i,:,:,0],0,1))*255
						imsave('prediction/' +str(step)+ '_p_a_c_'+str(i)+'.png',
							np.stack((preds3[i,:,:,0], preds2[i,:,:,0], filler),axis=2) )

						imsave('prediction/' +str(step)+ '_l_a_c_'+str(i)+'.png',
							np.stack((label_angles[i,:,:,1],label_angles[i,:,:,0],filler),axis=2) )

			else:
				loss_value,lossa,loss_sem,loss_c, _ = self.sess.run([self.reduced_loss,self.ins_a_loss,self.sem_loss,self.ins_c_loss, self.train_op],
					feed_dict=feed_dict)
			duration = time.time() - start_time
			print(self.conf.print_color + 'step {:d} \t total loss = {:.3f},sem. loss = {:.3f},ang. loss = {:.5f},cent. loss = {:.5f},({:.3f} sec/step)'.format(step, loss_value,loss_sem,lossa,loss_c, duration) + normal_color)
			write_log('{:d}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(step, loss_value,lossa,loss_sem,loss_c), self.conf.logfile)

		# finish
		self.coord.request_stop()
		self.coord.join(threads)

	def train_setup(self):
		tf.set_random_seed(self.conf.random_seed)

		# Create queue coordinator.
		self.coord = tf.train.Coordinator()
		if self.conf.pretrain_file == 'none':
			self.conf.pretrain_file = None
		# Input size!
		input_size = (self.conf.input_height, self.conf.input_width)

		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.data_list,
				input_size,
				self.conf.random_scale,
				self.conf.random_mirror,
				self.conf.random_flip,
				self.conf.ignore_label,
				IMG_MEAN,
				self.coord)
			self.image_batch, self.label_batch= reader.dequeue(self.conf.batch_size)

		# Create network
		if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50, or deeplab")
			sys.exit(-1)
		elif self.conf.encoder_name == 'deeplab':
			net = Deeplab_v2(self.image_batch, self.conf.num_classes, True)
			# Variables that load from pre-trained model.
			restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
			# Trainable Variables
			all_trainable = tf.trainable_variables()
			# Fine-tune part
			encoder_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
			# Decoder part
			decoder_trainable = [v for v in all_trainable if 'fc' in v.name]
		else:
			net = ResNet_segmentation(self.image_batch, self.conf.num_classes, True, self.conf.encoder_name)
			# Variables that load from pre-trained model.
			#restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name]
			if self.conf.pretrain_file is not None:
				reader = tf.train.NewCheckpointReader(self.conf.pretrain_file)
				restore_var = dict()
				for v in tf.trainable_variables():
					tensor_name = v.name.split(':')[0]
					if reader.has_tensor(tensor_name):
						#print('has tensor ', tensor_name)
						if 'ad_hoc_debug' in tensor_name:
							pass
						else:
							restore_var[tensor_name] = v
			else:
				restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name]

			# Trainable Variables
			all_trainable = tf.trainable_variables()
			# Fine-tune part
			encoder_trainable = [v for v in all_trainable if 'resnet_v1' in v.name] # lr * 1.0
			# Decoder part
			decoder_trainable = [v for v in all_trainable if 'decoder' in v.name]

		decoder_w_trainable = [v for v in decoder_trainable if 'weights' in v.name or 'gamma' in v.name] # lr * 10.0
		decoder_b_trainable = [v for v in decoder_trainable if 'biases' in v.name or 'beta' in v.name] # lr * 20.0
		# Check
		assert(len(all_trainable) == len(decoder_trainable) + len(encoder_trainable))
		assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))

		# Network raw output
		triple_out=net.outputs
		raw_output = triple_out[0] # [batch_size, h, w, 21]
		center_output = triple_out[2] # [batch_size, h, w, 21]
		angle_output = triple_out[1] # [batch_size, h, w, 21]
		print('semantic output shape: ',raw_output.shape)
		print('center prediction shape: ',center_output.shape)
		print('angular distance shape: ',angle_output.shape)


		# Output size
		output_shape = tf.shape(raw_output)
		output_size = (output_shape[1], output_shape[2])

		# Groud Truth semantic masks
		label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes, one_hot=False)
		#label_view=tf.reshape(label_proc,[self.conf.batch_size,output_shape[1],output_shape[2]])
		#self.label_view=label_view

		# Convert image semantic masks to instance masks
		label_angles=tf.py_func(self.angle_transform,[label_proc,0.3], tf.float64)
		self.label_angles=tf.image.resize_bilinear(label_angles, input_size)



		raw_gt = tf.reshape(label_proc, [-1,])
		raw_gt_a = tf.cast(tf.reshape(label_angles[:,:,:,0], [-1,1]),tf.float32)
		raw_gt_c = tf.cast(tf.reshape(label_angles[:,:,:,1], [-1,1]),tf.float32)

		indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)

		gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)


		raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
		prediction = tf.gather(raw_prediction, indices)

		raw_angle = tf.cast(tf.reshape(angle_output, [-1, 1]),tf.float32)
		raw_center = tf.cast(tf.reshape(center_output, [-1, 1]),tf.float32)
		self.c1=center_output
		self.c2=label_angles[:,:,:,1]

		# Pixel-wise softmax_cross_entropy loss
		self.sem_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
		self.ins_c_loss = tf.reduce_mean(tf.pow(raw_center - raw_gt_c,2))*10
		self.ins_a_loss = tf.reduce_mean(tf.abs(raw_angle - raw_gt_a))


		#loss=-tf.reduce_mean((gt*tf.log(tf.nn.softmax(prediction))*my_distance_maps))
		#loss=-((tf.cast(gt,tf.float32)*tf.log(tf.nn.softmax(prediction))))
		# L2 regularization
		l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
		# Loss function
		self.reduced_loss =self.sem_loss + tf.add_n(l2_losses)+self.ins_a_loss+self.ins_c_loss

		# Define optimizers
		# 'poly' learning rate
		base_lr = tf.constant(self.conf.learning_rate)
		self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
		learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.conf.num_steps), self.conf.power))
		# We have several optimizers here in order to handle the different lr_mult
		# which is a kind of parameters in Caffe. This controls the actual lr for each
		# layer.
		opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
		opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
		opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
		# To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
		# Instead, we separate the steps compute_grads+update_params.
		# Compute grads
		grads = tf.gradients(self.reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable)
		grads_encoder = grads[:len(encoder_trainable)]
		grads_decoder_w = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_w_trainable))]
		grads_decoder_b = grads[(len(encoder_trainable) + len(decoder_w_trainable)):]
		# Update params
		train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable))
		train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, decoder_w_trainable))
		train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, decoder_b_trainable))
		# Finally, get the train_op!
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
		with tf.control_dependencies(update_ops):
			self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

		# Saver for storing checkpoints of the model
		self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

		# Loader for loading the pre-trained model
		self.loader = tf.train.Saver(var_list=restore_var)

		# Training summary
		# Processed predictions: for visualisation.

		raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
		raw_output_up = tf.argmax(raw_output_up, axis=3)

		self.pred = tf.expand_dims(raw_output_up, dim=3)



		raw_output_up_a = tf.image.resize_bilinear(angle_output, input_size)
		self.pred_2 = raw_output_up_a

		raw_output_up_c = tf.image.resize_bilinear(center_output, input_size)
		self.pred_3 = raw_output_up_c
		# Image summary.

		images_summary = tf.py_func(inv_preprocess, [self.image_batch, 2, IMG_MEAN], tf.uint8)
		self.images_summary=images_summary
		labels_summary = tf.py_func(decode_labels, [self.label_batch, 2, self.conf.num_classes], tf.uint8)
		self.labels_summary=labels_summary
		preds_summary = tf.py_func(decode_labels, [self.pred, 2, self.conf.num_classes], tf.uint8)
		self.preds_summary=preds_summary
		self.total_summary = tf.summary.image('images',
			tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
			max_outputs=2) # Concatenate row-wise.
		if not os.path.exists(self.conf.logdir):
			os.makedirs(self.conf.logdir)
		self.summary_writer = tf.summary.FileWriter(self.conf.logdir, graph=tf.get_default_graph())

	def save(self, saver, step):
		'''
		Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, saver, filename):
		'''
		Load trained weights.
		'''
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))


	def angle_transform(self,images,thresh):
		o1,o2,o3=np.shape(images)
		angles_out=np.zeros((self.conf.batch_size,o2,o3,2)).astype('float64')
		for imidx in range(self.conf.batch_size):
			if np.sum(images[imidx,:,:])==0:
				pass
			else:

				for classidx in range(0,self.conf.num_classes):
					pk_sv=np.zeros((o2,o3))
					mask=np.zeros((o2,o3))

					if classidx==0:
						continue
					elif classidx==1:
						continue
					else:
						im=images[imidx,:,:]

						mask[im==classidx]=1
						if np.sum(mask)==0:
							continue
						else:
							dt=distance_transform_edt(mask)
							'''
							[d1,d2]=np.shape(mask)

							[rr,cc]=np.meshgrid(range(d2),range(d1));

							r=cc-dt[1][0]
							c=rr-dt[1][1]
							r=np.divide(r,dt[0])
							c=np.divide(c,dt[0])
							r[mask==0]=0
							c[mask==0]=0
							r=(r-np.min(r))/(np.max(r)-np.min(r))
							c=(c-np.min(c))/(np.max(c)-np.min(c))
							r[mask==0]=0
							c[mask==0]=0
							angles_out[imidx,:,:,0]+=c
							angles_out[imidx,:,:,1]+=r
							'''
							angles_out[imidx,:,:,0]+=dt

							lab,num_regs=label(mask)

							for lr in range(1,num_regs+1):

								dts=np.copy(dt)
								dts[np.where(lab != lr)]=0
								pk_sv[np.where(dts>thresh*np.max(dts))]=1

							angles_out[imidx,:,:,1]+=pk_sv
		return angles_out
