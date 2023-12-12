

import pandas as pd
import json
import re	
from text_util import normalize
import pickle
from ast import literal_eval as le
import string
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from datetime import datetime
from keras.layers import Input, Embedding, GRU, LSTM, Dense, Lambda, LeakyReLU, Dense, Concatenate, Flatten
from keras.layers import TimeDistributed, Bidirectional

from keras.layers import Layer, Flatten, Lambda
from keras.models import Model
from keras import activations, initializers, constraints
from keras.regularizers import l1, l2, l1_l2
from keras_preprocessing.sequence import pad_sequences
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt
from collections import Counter
#<--get the reproducible results> 
seed_value= 0
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)



MAX_SENTENCE_LENGTH = 120
MAX_SENTENCE_COUNT = 50
MAX_COMS_COUNT = 150
MAX_COMS_LENGTH = 120
embedding_dim = embedding_dim = #choose embedding dimension from {50, 100, 200, 300}
batch_size = 16
Epochs = XXX
patience = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224



threshold_news = #0.02666666(=1/37.5) for ReCOVery and 0.016(=1/62.5) for MMCoVaR 
threshold_com = #0.0133(=1/75) for both the Datasets
threshold_image = #0.00125944584(=1/794) for both the Datasets

lr = # choose learning rate from {0.001, 0.0001, 0.00001}
pout = "repo_Fast/"


loss = 'binary_crossentropy'


METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)




if __name__ == "__main__":
	
	#=====Load All the stored data ==================
	encoder_decoder = tf.keras.models.load_model("../ReCoVery_reconstruct/model_image_recon_Aug_90")
	model_encoder = tf.keras.Model(inputs = encoder_decoder.inputs, outputs = encoder_decoder.get_layer('reshape_encoder').output)
	model_decoder = tf.keras.Model(inputs = encoder_decoder.get_layer('reshape_encoder').output, outputs = encoder_decoder.get_layer('decoder_out').output)	

	encoded_train_x = np.load('input/encoded_train_x.npy')
	encoded_train_c = np.load('input/encoded_train_c.npy')
	image_x = np.load('input/image_x.npy')	

	encoded_val_x = np.load('input/encoded_val_x.npy')
	encoded_val_c = np.load('input/encoded_val_c.npy')
	image_val = np.load('input/image_val.npy')
	
	val_y = np.load('input/val_y.npy')
	val_inverse_y = np.load('input/val_inverse_y.npy')


	train_y = np.load('input/train_y.npy')
	train_inverse_y = np.load('input/train_inverse_y.npy')

	embedding_matrix = np.load('input/embedding_matrix.npy')

	with open('input/word_index.pickle', 'rb') as f:
		word_index = pickle.load(f)

	with open('input/val_x.pickle', 'rb') as f:
		val_x = pickle.load(f)
	
	with open('input/val_c.pickle', 'rb') as f:
		val_c = pickle.load(f)


	with open('input/id_train.pickle', 'rb') as f:
		id_train = pickle.load(f)

	with open('input/id_test.pickle', 'rb') as f:
		id_test = pickle.load(f)


	print("done...data storing...")

		
	model = tf.keras.models.load_model("Model") #----Load Pretrained mEXACT Model-----
	

	news_embed_msls = np.load('news_embed.npy') #load the embedding of News text where each entry is either 1 or 0. here 1 indicates MS information
	com_embed_msls = np.load('com_embed.npy')
	image_embed_msls = np.load('image_embed.npy')

	

	R_news, R_news_attn, R_news_soft = model.get_layer('artificialThresholding_news').output[0], model.get_layer('artificialThresholding_news').output[1], model.get_layer('artificialThresholding_news').output[2] 
	R_comment, R_comment_attn, R_comment_soft = model.get_layer('artificialThresholding_com').output[0], model.get_layer('artificialThresholding_com').output[1], model.get_layer('artificialThresholding_com').output[2] 

	R_image, R_image_attn, R_image_soft = model.get_layer('artificialThresholding_image').output[0], model.get_layer('artificialThresholding_image').output[1], model.get_layer('artificialThresholding_image').output[1] 

	

	
	model_explain = Model(inputs = model.inputs, outputs = [R_news, R_news_attn, R_news_soft, R_comment, R_comment_attn, R_comment_soft, R_image, R_image_attn, R_image_soft])	

	
	R_news, R_news_attn,R_news_soft, R_comment, R_comment_attn, R_comment_soft, R_image, R_image_attn, R_image_soft = model_explain.predict([encoded_val_c,encoded_val_x,image_val]) 

	
	y_predicted = model.predict([encoded_val_c, encoded_val_x, image_val])
	y_predicted_main = y_predicted[0]
	y_predicted_inverse = y_predicted[1]
	y_predicted_main = y_predicted_main.flatten()
	y_predicted_inverse = y_predicted_inverse.flatten()
	
	y_pred_sigmoid = y_predicted_main
	y_predicted_main_inverse = y_predicted_inverse


	y_predicted_main = np.where(y_predicted_main > 0.5, 1, 0)
	y_predicted_inverse = np.where(y_predicted_inverse > 0.5, 1, 0)
	
	y_predicted_main = y_predicted_main.tolist()
	y_predicted_inverse = y_predicted_inverse.tolist()	


	tot_count=0
	count=0
	for x,y,z in zip(y_predicted_main, y_predicted_inverse,val_y):
		if x == z:
			if x == 0:
				tot_count = tot_count+1		
				if x != y:
					count = count + 1
	print(f"count of toggling sucess: {count}")
	
	print(f"rate of toggling sucess : {float(count/tot_count)}")

	
		
	news_msL, com_msL, news_lsL, com_lsL, groundL, predL,predL_inverse, pred_trueL, pred_fakeL, absL = [], [], [], [], [], [], [], [], [], []
	
	for i,(x,y,z,p,q,r,s,t,u) in enumerate(zip(y_predicted_main, y_predicted_inverse,val_y,val_x,val_c,y_pred_sigmoid, news_embed_msls, com_embed_msls, image_embed_msls)):
		
		loc_news = np.where(s == 1)[0].tolist()
		loc_com = np.where(t == 1)[0].tolist()
		loc_image = np.where(u==1)[0].tolist()

		news_msL.append([p[loc] for loc in loc_news])
		news_lsL.append([p[i] for i in range(len(p)) if i not in loc_news])
					
		com_msL.append([q[loc] for loc in loc_com])
		com_lsL.append([q[j] for j in range(len(q)) if j not in loc_com])

		groundL.append(z.item())
		predL.append(x)
		predL_inverse.append(y)	
		fake = r.item()
		true = 1.0 - (r.item())
					
		pred_trueL.append(true)
		pred_fakeL.append(fake)			
		absL.append(abs(true-fake))
		
					
	
	df_explain = pd.DataFrame()


	df_explain["testID"], df_explain["news_ms"],df_explain["news_ls"], df_explain["com_ms"],df_explain["com_ls"], df_explain["GT"],df_explain["Pred"],df_explain["Pred_inverse"], df_explain["prob_T"], df_explain["prob_F"], df_explain["absolute"] = id_test, news_msL, news_lsL, com_msL, com_lsL, groundL, predL, predL_inverse, pred_trueL, pred_fakeL, absL


	df_explain = df_explain.sort_values(by=['absolute'], ascending=True)

	df_explain.to_csv("explainable_samples.csv", index = False)


	#==========================================
	ax1, ax2, ax3 = R_image.shape #(428, 784, 100)
	from math import sqrt
	dim = int(sqrt(ax2))	#dim: 28
	
	R_image = R_image.reshape(ax1, dim,dim, ax3) #(428,28,28,100)
	print("==now===")
	print(f"shape of R_image: {R_image.shape}")#shape of R_image: (455, 28,28, 100)
	
	
	image_val_demo_ls = np.copy(image_val)
	image_val_demo_ms = np.copy(image_val)
	
	ax1, ax2, ax3, ax4 = image_val_demo_ls.shape
	image_val_demo_ls = image_val_demo_ls.reshape(ax1, ax2*ax3, ax4) #(455,784,100)
	image_val_demo_ms = image_val_demo_ms.reshape(ax1, ax2*ax3, ax4) #(455,784,100)
	print(f"image_val_demo_ls: {image_val_demo_ls.shape}")
	print(f"image_val_demo_ms: {image_val_demo_ms.shape}")

	for i in range(R_image_attn.shape[0]):
		loc_image = np.where(R_image_attn[i] == 0.0)[0].tolist()
		for idx in loc_image:
			image_val_demo_ls[i][idx] = 0.0		
		for j in range(image_val_demo_ms.shape[1]):
			if j not in loc_image:
				image_val_demo_ms[i][j] = 0.0			



	image_val_demo_ls = image_val_demo_ls.reshape(ax1, ax2,ax3, ax4) #(M, 28,28, 100) #M is size of test data
	image_val_demo_ms = image_val_demo_ms.reshape(ax1, ax2,ax3, ax4) #(M, 28,28, 100)

	recon_image =  model_decoder.predict(image_val)

	explain_image_ls = model_decoder.predict(image_val_demo_ls)
	explain_image_ms = model_decoder.predict(image_val_demo_ms)
	
		


	for i in range(R_image_attn.shape[0]):
		plt.imshow(recon_image[i])
		plt.savefig('ground_truth_demo/%s.jpg'%str(id_test[i]), bbox_inches='tight', pad_inches = 0)



		plt.imshow(explain_image_ms[i])
		plt.savefig('explainable_ms_demo/%s.jpg'%str(id_test[i]), bbox_inches='tight', pad_inches = 0)

		
		plt.imshow(explain_image_ls[i])
		plt.savefig('explainable_ls_demo/%s.jpg'%str(id_test[i]), bbox_inches='tight', pad_inches = 0)
	
