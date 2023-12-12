import keras 
from keras.layers import Input, Embedding, GRU, LSTM, Dense, Lambda, LeakyReLU, Dense, Concatenate, Flatten
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Layer, Flatten, Lambda
from keras.models import Model
from keras import activations, initializers, constraints
from keras.regularizers import l1, l2, l1_l2

from keras_preprocessing.sequence import pad_sequences#from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout

#<--get the reproducible results> 
seed_value= 0
import os
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

opt=tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.5) #choose learning rate from {0.001, 0.0001, 0.00001}, here we choose 0.001 for optimal result

units_threshold = 100



MAX_SENTENCE_LENGTH = 120
MAX_SENTENCE_COUNT = 50
MAX_COMS_COUNT = 150
MAX_COMS_LENGTH = 120
embedding_dim = #choose embedding dimension from {50, 100, 200, 300}
image_dim = 784
loss = 'binary_crossentropy'

METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]

threshold_news = #0.02666666(=1/37.5) for ReCOVery and 0.016(=1/62.5) for MMCoVaR 
threshold_com = #0.0133(=1/75) for both the Datasets
threshold_image = #0.00125944584(=1/794) for both the Datasets



class AttLayer(Layer):

	def __init__(self, **kwargs):
		super(AttLayer, self).__init__(**kwargs)
		self.init = initializers.get('normal')
		self.supports_masking = True
		self.attention_dim = 100

	def build(self, input_shape):
		assert len(input_shape) == 3, "Input shape must be 3"
		self.W = K.variable(self.init((input_shape[-1], self.attention_dim))) #(GRU_units, 100) = (200,100)
		self.b = K.variable(self.init((self.attention_dim,))) #(100, )
		self.u = K.variable(self.init((self.attention_dim, 1))) #(100,1)
		#self._trainable_weights = [self.W, self.b, self.u]
		super(AttLayer, self).build(input_shape)

	def compute_mask(self, inputs, mask=None):
		return mask

	def call(self, x, mask=None):
		# size of x :[batch_size, MAX_SENTENCE_LENGTH, GRU_units] (None, 120, 200)
		# size of u :[batch_size, attention_dim] (None, 100)
		# uit = tanh(xW+b)
		
		uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b)) #it means K.dot(x,self.W)) =(None, 120, 100) and K.bias_add(K.dot(x, self.W), self.b) = (None, 120, 100)
		ait = K.dot(uit, self.u) #(None, 120, 1)
		ait = K.squeeze(ait, -1) #(None, 120)
		ait = K.exp(ait) #(None, 120)
		
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			ait *= K.cast(mask, K.floatx())
		print(ait)
		ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx()) # (None, 120)  where K.sum(ait, axis=1, keepdims=True) is (None,1)
		ait = K.expand_dims(ait) #(None, 120, 1)
		weighted_input = x * ait #(None, 120, 200)
		output = K.sum(weighted_input, axis=1) #(None, 200)
		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1]) #(None, 200)









class artificialThresholding(tf.keras.layers.Layer):
	def __init__(self, threshold, category, units = 100,kernel_initializer=tf.keras.initializers.glorot_uniform(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), kernel_constraint=None, use_bias=True, bias_initializer=tf.keras.initializers.glorot_uniform(), bias_regularizer=None, bias_constraint=None, **kwargs):
		
		self.units = units
		self.threshold = threshold	
		self.category = category	
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.use_bias = use_bias
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)


		super(artificialThresholding, self).__init__(**kwargs)

	def build(self, input_shape):

		self._kernel = self.add_weight(name = 'kernel', shape=[input_shape[-1], self.units], initializer=self.kernel_initializer,regularizer=self.kernel_regularizer,constraint=self.kernel_constraint, dtype=self.dtype, trainable=True) #(200, 100)
				
		if self.use_bias:
			self._bias = self.add_weight(name = 'bias', shape=[self.units,], initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
			self._bias_dash = self.add_weight(name = 'bias_dash', shape=[self.units,], initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
		


	def call(self, inputs):
		print(f"start..............{inputs.shape}") #(None, 50, 200) or (None, 150, 200) or (one,784,100)
		if self.category == '1' or self.category == '2':		
			t1 = tf.math.reduce_sum(inputs, axis =2) #(None, 50) or (N,150)
			print(f"after reduce_sum: {t1.shape}") 
				
		if self.category == '3':		
			print("============== Image =====================")
						
			t1 = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(inputs) #(None,784)
			print(f"after globalaveragepooling: {t1.shape}") 


		t1 = tf.keras.activations.softmax(t1, axis=1)#(None, 50) or (N,150) or (None,784)
		t1_soft = t1

		k = self.threshold
		
		mask=tf.less(t1, k * tf.ones_like(t1)) #(None, 50) or (N,150) or (None,784)
			
		t1 = tf.multiply(t1, tf.cast(mask, t1.dtype)) #(None, 50) or (N,150, 200)
		res_thresholding = tf.multiply(inputs, t1[:, :, tf.newaxis]) #(None, 50, 200) or (None, 150, 200) or (None,784,100)
		
		return res_thresholding, t1, t1_soft

	def compute_output_shape(self,input_shape):
		l = [(input_shape[0], input_shape[1], input_shape[-1]), (input_shape[0], input_shape[1])] 			 	
		return l	


	def compute_mask(self, inputs, mask=None):
		return [None,None]





class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs1, num_outputs2,num_outputs3,num_outputs4, kernel_initializer=tf.keras.initializers.glorot_uniform(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), kernel_constraint=None, use_bias=True, bias_initializer=tf.keras.initializers.glorot_uniform(), bias_regularizer=None, bias_constraint=None,  **kwargs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs1 = num_outputs1
		self.num_outputs2 = num_outputs2
		self.num_outputs3 = num_outputs3
		self.num_outputs4 = num_outputs4
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)
		super(MyDenseLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		
		self.kernel1 = self.add_weight("kernel_dense1",shape=[int(input_shape[-1]),self.num_outputs1], trainable=True)
		self.kernel2 = self.add_weight("kernel_dense2",shape=[self.num_outputs1,self.num_outputs2], trainable=True)
		self.kernel3 = self.add_weight("kernel_dense3",shape=[self.num_outputs2,self.num_outputs3], trainable=True)
		self.kernel4 = self.add_weight("kernel_dense4",shape=[self.num_outputs3,self.num_outputs4], trainable=True)
		

		self.bias1 = self.add_weight('bias1',shape = [self.kernel1.shape[-1],], initializer='zeros', trainable=True)
		self.bias2 = self.add_weight('bias2',shape = [self.kernel2.shape[-1],], initializer='zeros', trainable=True)
		self.bias3 = self.add_weight('bias3',shape = [self.kernel3.shape[-1],], initializer='zeros', trainable=True)
		self.bias4 = self.add_weight('bias4',shape = [self.kernel4.shape[-1],], initializer='zeros', trainable=True)
		

	def call(self, inputs):
		layer_ReLU = tf.keras.layers.ReLU()
		
		dense1 = tf.matmul(inputs, self.kernel1)
		dense1 = dense1 + self.bias1
		dense1 = layer_ReLU(dense1)
		dense1 = Dropout(0.2)(dense1)
		
		dense2 = tf.matmul(dense1, self.kernel2)
		dense2 = dense2 + self.bias2
		dense2 = layer_ReLU(dense2)
		dense2 = Dropout(0.2)(dense2)
		
		
		dense3 = tf.matmul(dense2, self.kernel3)
		dense3 = dense3 + self.bias3
		dense3 = layer_ReLU(dense3)
		dense3 = Dropout(0.2)(dense3)		


		dense4 = tf.matmul(dense3, self.kernel4)
		dense4 = dense4 + self.bias4
		dense4 = layer_ReLU(dense4)
		dense4 = Dropout(0.2)(dense4)
		
		return dense4






def build_NeuralNet(embedding_matrix, word_index):
	#<====News Neural Network====>
	
	#<--First we prepare embedding layers for news and comments-->

	embedding_layer_news = Embedding(len(word_index) + 1, embedding_dim, weights = [embedding_matrix], input_length = MAX_SENTENCE_LENGTH, trainable = True, mask_zero = True)	
	
	embedding_layer_comments = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length = MAX_COMS_LENGTH, trainable = True, mask_zero=True)	

	#<--work on news--->
	news_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')#(None, 120)
	embedded_news = embedding_layer_news(news_input) #(None, 120, 100)
	news_GRU = Bidirectional(GRU(100, return_sequences=True), name='word_GRU')(embedded_news) #(None, 120, 200)
	news_att = AttLayer(name='word_attention')(news_GRU)  #(None, 200)
	print("news_input: ",news_input.shape) 
	print("embedded_news: ", embedded_news.shape) 
	print("news_GRU: ", news_GRU.shape)
		
	sentEncoder = Model(news_input, news_att)
	
	print(sentEncoder.summary())
	
	content_input = Input(shape=(MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype='int32');print("shape of content_input: ",content_input.shape) #(None, 50, 120)
	content_encoder = TimeDistributed(sentEncoder, name='time_distributed_news')(content_input);print("shape of content_encoder: ",content_encoder.shape) #(None, 50, 200)
	
	news_content_sentence_level_encoder = Model(content_input, content_encoder)
	print(news_content_sentence_level_encoder.summary())
		
	#<---work on comments--->
	
	comment_input = Input(shape=(MAX_COMS_LENGTH,), dtype='int32') #(None, 120)
	com_embedded_sequences = embedding_layer_comments(comment_input) #(None, 120, 100)
	c_GRU = Bidirectional(GRU(100, return_sequences=True), name='comment_lstm')(com_embedded_sequences) #(None, 120, 200)
	c_att = AttLayer(name='comment_word_attention')(c_GRU) #(None, 200)
	
	print("com_embedded_sequences: ",com_embedded_sequences.shape)
	print("c_att: ",c_att.shape)
	comEncoder = Model(comment_input, c_att, name='comment_word_level_encoder')
	
	all_comment_input = Input(shape=(MAX_COMS_COUNT, MAX_COMS_LENGTH), dtype='int32') #(None, 150, 120)
	all_comment_encoder = TimeDistributed(comEncoder, name='time_distributed_comments')(all_comment_input) ##(None, 150, 200)
	print("all_comment_encoder: ",all_comment_encoder.shape)
	comment_sequence_encoder = Model(all_comment_input, all_comment_encoder)
	
	concat_res = Concatenate(axis=1)([content_encoder, all_comment_encoder]) #(None, 200, 200) where all_comment_encoder is (None, 150, 200) and content_encoder is (None, 50, 200)      
	print("concat_res.shape: ",concat_res.shape)

	ax1, ax2, ax3 = concat_res.get_shape().as_list()	
	concat_news_com = tf.reshape(concat_res, [-1, ax2*ax3])
	print(f"concat_news_com.shape: {concat_news_com.shape}")
	
	print("==============Image===============")
	
	#<--For images--->
	image_input = Input((28, 28, 100))
	ax1, ax2, ax3, ax4 = image_input.get_shape().as_list()		
	x = tf.reshape(image_input, [-1, ax2*ax3*ax4])
	print(f"after reshape: {x.shape}")#(None, 78400)
	
	concat_all = Concatenate(axis=1)([concat_news_com, x])
	print("res after final conacat: ", concat_all.shape)

	#<==Prepare for softmax Thresholding===>
	image_embed = tf.reshape(image_input, [-1, ax2*ax3, ax4]) # (None, 784, 100) where image_input is of shape (None, 28,28,100)
	print(f"image_embed.shape: {image_embed.shape}")	
				
	cat_news, cat_com, cat_image = '1', '2', '3' # We call cat_news as category of news and so on.

	R_news, R_news_no_relu, R_news_soft = artificialThresholding(threshold_news, cat_news, units_threshold,name = 'artificialThresholding_news')(content_encoder)  
	R_comment, R_comment_no_relu, R_comment_soft = artificialThresholding(threshold_com, cat_com, units_threshold,name = 'artificialThresholding_com')(all_comment_encoder) 
	R_image, R_image_no_relu, R_image_soft = artificialThresholding(threshold_image, cat_image,units_threshold, name = 'artificialThresholding_image')(image_embed) 

	concat_V_news_comment = Concatenate(axis=1)([R_news, R_comment]);print(concat_V_news_comment.shape)
	
	ax1, ax2, ax3 = concat_V_news_comment.get_shape().as_list()
	concat_V_news_comment = tf.reshape(concat_V_news_comment, [-1, ax2*ax3])

	ax1, ax2, ax3 = R_image.get_shape().as_list()
	R_image = tf.reshape(R_image, [-1, ax2*ax3])

	concat_V_news_comment_image = Concatenate(axis=1)([concat_V_news_comment, R_image])
		
	#<===Proceed the remaining part of NN===>

	layer_dense = MyDenseLayer(1024, 512,256,128)
	concat_all = layer_dense(concat_all)		
	concat_V_news_comment_image = layer_dense(concat_V_news_comment_image)
	
	preds = Dense(1, activation='sigmoid', name="main_output")(concat_all)
	preds_inverse = Dense(1, activation='sigmoid', name="inverse_output")(concat_V_news_comment_image)## adv before final x'=0
	
	model = Model(inputs=[all_comment_input, content_input, image_input], outputs=[preds,preds_inverse])#+list_min_max_Loss)
	
	print(model.summary())
	
	loss = ['binary_crossentropy']+['binary_crossentropy']
	
	loss_weights = [1.4]+[1.4]
	

	
	model.compile(optimizer=opt,
	              loss=loss,
	              loss_weights=loss_weights,
		      metrics={'main_output': 'accuracy', 'inverse_output': 'accuracy'})

	return model	

