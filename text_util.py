#<===How to install Spacy==>
#https://spacy.io/usage

#<===Test models===>
#https://spacy.io/models/en



#<===Use PunktSentenceToeizer====>
#https://datascience.stackexchange.com/questions/87793/converting-paragraphs-into-sentences
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import string
import re
import nltk
import pandas as pd
from nltk import tokenize


def performCleaning(news):
	news = news.strip()
	news = re.sub(r'http\S+', '', news) #remove hyperlinks
	
	
	STOP_WORDS = ['the', 'a', 'an']	
	news = ' '.join([word for word in news.split() if word not in STOP_WORDS and word not in string.punctuation])	
	news = re.sub('\s+',' ',news)	#remove whitespaces https://stackoverflow.com/questions/10711116/strip-spaces-tabs-newlines-python

	return news



def normalize(text):
	sentencesL = tokenize.sent_tokenize(text)
	sentencesL = [performCleaning(sentence) for sentence in sentencesL]
	
	return sentencesL
	


