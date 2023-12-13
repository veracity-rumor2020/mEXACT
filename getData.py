
import pickle
import pandas as pd
import json
import re	
from text_util import normalize
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
#from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences#from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import getModel as gM
import matplotlib.pyplot as plt
from collections import Counter
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



MAX_SENTENCE_LENGTH = 120
MAX_SENTENCE_COUNT = 50
MAX_COMS_COUNT = 150
MAX_COMS_LENGTH = 120
embedding_dim = #choose embedding dimension from {50, 100, 200, 300}
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

trainStatsFile = pout+"epoch_"+str(Epochs)+"_"+str(lr)+"_"+str(threshold_news)+"_"+str(threshold_com)+"_"+str(threshold_image)+"_"+str(embedding_dim)+"_"+str(batch_size)+".csv"
calmetricFile = pout+"epoch_"+str(Epochs)+"_"+str(lr)+"_"+str(threshold_news)+"_"+str(threshold_com)+"_"+str(threshold_image)+"_"+str(embedding_dim)+"_"+str(batch_size)+".npz"

loss = 'binary_crossentropy'


METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)


image_base_path_all = "../../MMCoVaR/Image_news/"
image_data_all = os.listdir(image_base_path_all)
di_image_id_name_all = {}

for image_data in image_data_all:
	di_image_id_name_all[image_data.split(".")[0]] = image_data




def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def performCleaning(tweet):
	#https://www.kaggle.com/code/redwankarimsony/nlp-101-tweet-sentiment-analysis-preprocessing/notebook
	tweet = re.sub(r'\bRT\b', '', tweet) #remove RT
	tweet = re.sub(r'#', '', tweet) #remove hashtags
	tweet = re.sub(r'@', '', tweet) #remove mentions
	
	#tweet =  re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	tweet = re.sub(r'http\S+', '', tweet) #remove hyperlinks
	tweet = re.sub(r'[0-9]', '', tweet) # remove numbers
	
	tweet = re.sub("'", "", tweet)
	
	
	STOP_WORDS = ['the', 'a', 'an']	
	tweet = ' '.join([word for word in tweet.split() if word not in STOP_WORDS and word not in string.punctuation])

	'''	
	#<==remove unwanted ... from text ===>	
	x = tweet.split('.')
	y = [ele.strip() for ele in x if ele]
	tweet = ' '.join(y)
	#<===================================>
	tweet = tweet.replace(',','') #remove comma
	
	
	#<====Lemmatization===>
	tweet_tokens = tokenizer.tokenize(tweet)
	lemmaL = []
	for word in tweet_tokens:
		lemmaL.append(wordnet_lemmatizer.lemmatize(word))
	tweet = ' '.join(lemmaL)
	'''	
	return tweet

def fit_on_news_and_comments(train_x, train_c, val_x, val_c):
	texts = []
	texts.extend(train_x)
	texts.extend(val_x)
	comments = []
	comments.extend(train_c)
	comments.extend(val_c)
	tokenizer = Tokenizer(num_words=20000)
	all_text = []
	all_sentences = []
	for text in texts:
		for sentence in text:
			all_sentences.append(sentence)

	all_comments = []
	for com in comments:
		for sentence in com:
			all_comments.append(sentence)

	all_text.extend(all_comments)
	all_text.extend(all_sentences)
	tokenizer.fit_on_texts(all_text)
	VOCABULARY_SIZE = len(tokenizer.word_index) + 1
	#print(" len of all_text:", len(all_text))	
	#print("vocab_size: ", VOCABULARY_SIZE)

	'''
	total_sen_len = sum([len(ele) for ele in train_x]) + sum([len(ele) for ele in val_x])
	print("total_sen_len: ", total_sen_len)
	print("len(all_sentences): ",len(all_sentences))

	total_com_len = sum([len(ele) for ele in train_c]) + sum([len(ele) for ele in val_c])
	print("total_com_len: ", total_com_len)
	print("len(all_comments): ",len(all_comments))
	'''
	
	reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}

	return VOCABULARY_SIZE, tokenizer, reverse_word_index


def build_Embedding_Matrix(glove, t, aff_dim=80):
	
	embeddings_index = {}
	f = open(glove)   
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	print('Loaded %s word vectors.' % len(embeddings_index))
	word_index = t.word_index
	embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	#return embedding_matrix

	print("shape of embedding_matrix: ", embedding_matrix.shape)
	
	return embedding_matrix, word_index

	


def _encode_texts(texts, t):
	encoded_texts = np.zeros((len(texts), MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH), dtype='int32')
	for i, text in enumerate(texts):
		encoded_text = np.array(pad_sequences(t.texts_to_sequences(text),maxlen=MAX_SENTENCE_LENGTH, padding='post', truncating='post', value=0))[:MAX_SENTENCE_COUNT]
		encoded_texts[i][:len(encoded_text)] = encoded_text

	return encoded_texts


def _encode_comments(comments, t):

	encoded_texts = np.zeros((len(comments), MAX_COMS_COUNT, MAX_COMS_LENGTH), dtype='int32')
	for i, text in enumerate(comments):
		encoded_text = np.array(pad_sequences(t.texts_to_sequences(text),maxlen=MAX_COMS_LENGTH, padding='post', truncating='post', value=0))[:MAX_COMS_COUNT]
		encoded_texts[i][:len(encoded_text)] = encoded_text

	return encoded_texts




if __name__ == "__main__":
	
	df = pd.read_csv("dataset.csv", delimiter = ",") #Read ReCOVery or MMCoVaR dataset	
	
	df['usersId'] =  df['usersId'].apply(lambda x:le(x))
	df['usersText'] =  df['usersText'].apply(lambda x:le(x))
	df_news = df[['news_id', 'image', 'body_text', 'reliability']]
	#reliability label of the news article (1 = reliable/true, 0 = unreliable/fake)
	df_news['reliability_inverse'] = df_news['reliability'].apply(lambda x:(1-x) if x!=1 else 1)
	df_comments = df[['news_id','usersId', 'usersText']]
	df_comments["usersText_clean"] = df_comments["usersText"].apply(lambda x:[performCleaning(ele) for ele in x]) 

	#<----Dealing with Images--->
	idL = df["news_id"].tolist()
	idL = [str(id_) for id_ in idL]	
	absentL = []
	for id_ in idL:
		if id_ not in list(di_image_id_name_all.keys()):
			absentL.append(id_)
	
	
	training_data = []
	for i, id_ in enumerate(idL):
		if id_ in list(di_image_id_name_all.keys()):
			file_loc = os.path.join(image_base_path_all,di_image_id_name_all[id_])
			if os.path.exists(file_loc):
				img_array = cv2.imread(file_loc)#, cv2.IMREAD_GRAYSCALE)
				if img_array is None:
					print(i,": ",id_)
					print(file_loc)					
					new_array = np.random.rand(224,224,3)			
	

				else:		
					new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))#;print("shape of img_array: ",new_array.shape);input("............................")
				
				training_data.append(new_array)				
				
		else:
			training_data.append(np.random.rand(224,224,3))

	
	print("Testing data size: ",len(training_data))#(1712, 224, 224, 3)	
	test_x=np.array(training_data)
	test_x=test_x/255.0
	print(test_x.shape)
	#======Load Pretrained Autoencoder model for image reconstruction===========
	encoder_decoder = tf.keras.models.load_model("model_image_recon_Aug_90") #Go under "Autoencoder_ReCOVery" director and unzip "model_image_recon_Aug_90.zip" file for loading autoencoder trained on ReCOVery dataset. For MMCoVaR check "Autoencoder_MMCoVaR" directory. 
	model_encoder = tf.keras.Model(inputs = encoder_decoder.inputs, outputs = encoder_decoder.get_layer('reshape_encoder').output)
	model_decoder = tf.keras.Model(inputs = encoder_decoder.get_layer('reshape_encoder').output, outputs = encoder_decoder.get_layer('decoder_out').output)
	
	print(model_encoder.summary())	
	print(model_decoder.summary())
	
	test_xZ = model_encoder.predict(test_x)
	
	print(f"toal Z.shape : {test_xZ.shape}") # Get total Z from trained autoencoder	(N, 28, 28, 100) where N is #training samples

	#<=== create News data and their labels ====>
	VALIDATION_SPLIT = 0.25	
	contents = []
	labels = []
	ids = []
	
	for i, row in df_news.iterrows():
		text = row["body_text"]	
		text = clean_str(text)
		sentences = normalize(text)
		contents.append(sentences)
		ids.append(row['news_id'])
		labels.append(row['reliability'])
		


	labels_inverse = df_news['reliability_inverse'].tolist()
	labels = np.asarray(labels);print(labels)
	labels_inverse = np.asarray(labels_inverse);print(labels_inverse)
		
	#<=== load user comments or tweets ===>
	comments_text = df_comments["usersText_clean"].tolist()
	
	#<===Training and testing split===>

	id_train, id_test, train_x, val_x, train_y, val_y, train_c, val_c, image_x, image_val,train_inverse_y, val_inverse_y = train_test_split(ids,contents, labels, comments_text,test_xZ,labels_inverse, test_size=VALIDATION_SPLIT, random_state=42,stratify=labels)
	

	print("===Start Work on Embedding===")
	VOCABULARY_SIZE, t, reverse_word_index = fit_on_news_and_comments(train_x, train_c, val_x, val_c)
	#choose embedding dimension from {glove.6B.50d.txt,glove.6B.100d.txt,glove.6B.200d.txt,glove.6B.300d.txt}
	embedding_matrix, word_index = build_Embedding_Matrix("embedding/glove.6B/glove.6B.100d.txt", t, aff_dim=80) #here we choose 100, for example. 

		
	encoded_train_x = _encode_texts(train_x,t)
	encoded_val_x = _encode_texts(val_x,t)
	encoded_train_c = _encode_comments(train_c,t)
	encoded_val_c = _encode_comments(val_c,t)
	
		
	model = gM.build_NeuralNet(embedding_matrix, word_index)	
	
	
		
	history = model.fit([encoded_train_c, encoded_train_x, image_x], y=[train_y,train_inverse_y], validation_split=0.2, batch_size=batch_size, epochs=Epochs, verbose=2)#, callbacks=[callback])
	
	model.save("Model_"+str(Epochs)+"_"+str(lr)+"_"+str(threshold_news)+"_"+str(threshold_com)+"_"+str(threshold_image)+"_"+str(embedding_dim)+"_"+str(batch_size)+"_dense4"+"_"+str(recon))	
	
	y_predicted = model.predict([encoded_val_c, encoded_val_x, image_val])

	y_predicted_main = y_predicted[0]
	y_predicted_inverse = y_predicted[1]
	
	#<---for first output terminal--->
	print("---for first output terminal---")
	y_predicted_main = y_predicted_main.flatten()
	y_predicted_main = np.where(y_predicted_main > 0.5, 1, 0)

	from sklearn.metrics import confusion_matrix, classification_report
	cm = confusion_matrix(val_y, y_predicted_main)
	print(cm)
	print(classification_report(val_y, y_predicted_main,digits=4))

		
	#<---for second output terminal--->	
	print("---for second output terminal---")
	y_predicted_inverse = y_predicted_inverse.flatten()
	y_predicted_inverse = np.where(y_predicted_inverse > 0.5, 1, 0)


	from sklearn.metrics import confusion_matrix, classification_report
	cm = confusion_matrix(val_inverse_y, y_predicted_inverse)
	print(cm)
	print(classification_report(val_inverse_y, y_predicted_inverse,digits=4))

	
	np.savez(calmetricFile, name1=val_y, name2=y_predicted_main, name3=val_inverse_y, name4=y_predicted_inverse)
