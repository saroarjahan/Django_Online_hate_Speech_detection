from django.views.generic import TemplateView
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.shortcuts import render, get_object_or_404,redirect
from sklearn import  metrics
from django.contrib import messages
import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from empath import Empath #for accuracy
from inscriptis import get_text
import urllib.request
import urllib3
import re
# import keras
# from keras.preprocessing import text
# from keras.preprocessing import sequence
# from keras import models
# from keras import layers
# from keras import optimizers


def aggression(request):
	
	if request.GET.get("q"):
		url = request.GET.get("q")
		html = urllib.request.urlopen(url).read().decode('utf-8')
		text = get_text(html)
		text=text.strip()
		text=text.split('.')

		comments = pd.read_csv('http://127.0.0.1:8000/static/aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations = pd.read_csv('http://127.0.0.1:8000/static/aggression_annotations.tsv',  sep = '\t')
		len(annotations['rev_id'].unique())

		# labels a comment as an atack if the majority of annoatators did so
		labels = annotations.groupby('rev_id')['aggression'].mean() > 0.5
		# print(labels)

		# join labels and comments
		comments['aggression'] = labels

		# =========================== Data Cleaning =====================

		# clean the text
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.lower())
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('.,[^a-zA-z0-9\s]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(' +',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(':',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('`',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('>',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('<',' ',x)))

		# remove newline and tab tokens
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

		hd=comments.query('aggression')['comment'].head()

		# keeping only training and test sets
		train_comments = comments.query("split=='train'")
		valid_comments = comments.query("split=='test'")

		# split the dataset into training and validation datasets 
		train_x, valid_x = train_comments['comment'], valid_comments['comment'], 
		train_y, valid_y = train_comments['aggression'], valid_comments['aggression']

		# # label encode the target variable 
		# encoder = preprocessing.LabelEncoder()
		# train_y = encoder.fit_transform(train_y)
		# valid_y = encoder.fit_transform(valid_y)

		# create a count vectorizer object 
		count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		# word level tf-idf
		tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		# ngram level tf-idf 
		tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
		# characters level tf-idf
		tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

		clf = Pipeline([
		    ('vect', tfidf_vect),
		    ('clf', LogisticRegression()),
		])
		clf = clf.fit(train_comments['comment'], train_comments['aggression'])
		auc = roc_auc_score(valid_comments['aggression'], clf.predict_proba(valid_comments['comment'])[:, 1])
		f1score = metrics.f1_score(valid_y, valid_comments['aggression'],average='weighted')
		#print('Test ROC AUC: %.3f' %auc)

		# load the pre-trained word-embedding vectors 
		# embeddings_index = {}
		# for i, line in enumerate(open('http://127.0.0.1:8000/static/wiki-news-300d-1M.vec', encoding="utf8")):
		# 	values = line.split()
		# 	embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

		# # create a tokenizer 
		# token = text.Tokenizer()
		# token.fit_on_texts(comments['comment'])
		# word_index = token.word_index

		# # convert text to sequence of tokens and pad them to ensure equal length vectors 
		# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
		# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

		# correctly classify nasty comment
		all_sen=""
		for sen in text:
		    x = clf.predict([sen])
		    if x==True:
		        all_sen=all_sen+sen


		# # correctly classify nasty comment
		# y=clf.predict(['People as stupid as you should not edit Wikipedia!'])
		# print(y)

		lexicon = Empath()
		x=lexicon.analyze(all_sen, categories=["negative_emotion", "positive_emotion"], normalize=True)




		if all_sen =='':
			all_sen="No Hate Sentence Found"
			x={'negative_emotion':0,"positive_emotion":0}

		appointments= {
		"query": x,
		'fs':f1score,
		'fscore':'Classifier F1 Score',
		"cl_name":": Linear Classifier on Word Level TF IDF Vectors",
		"text":all_sen,
		"auc":auc,
		"accuracy":"Classifier Accuracy :",
		"aggression":"Aggression",
		"caggression":"Aggression",
		"sentence":"Identified Agrresive Sentences :",
		"Liwc_pos":"Positve Emotion :",
		"Liwc_neg":"Negative Emotion :",
		
		}
		return render(request, 'index.html', appointments )
	else:
		appointments= {
		"caggression":"Aggression",
		}
		return render(request, 'index.html', appointments )


def attack(request):
	
	if request.GET.get("q"):
		url = request.GET.get("q")
		html = urllib.request.urlopen(url).read().decode('utf-8')
		text = get_text(html)
		text=text.strip()
		text=text.split('.')


		comments = pd.read_csv('http://127.0.0.1:8000/static/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations = pd.read_csv('http://127.0.0.1:8000/static/attack_annotations.tsv',  sep = '\t')
		len(annotations['rev_id'].unique())

		# labels a comment as an atack if the majority of annoatators did so
		labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
		# print(labels)

		# join labels and comments
		comments['attack'] = labels

		# =========================== Data Cleaning =====================

		# clean the text
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.lower())
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('.,[^a-zA-z0-9\s]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(' +',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(':',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('`',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('>',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('<',' ',x)))

		# remove newline and tab tokens
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

		hd=comments.query('attack')['comment'].head()

		# keeping only training and test sets
		train_comments = comments.query("split=='train'")
		valid_comments = comments.query("split=='test'")

		# split the dataset into training and validation datasets 
		train_x, valid_x = train_comments['comment'], valid_comments['comment'], 
		train_y, valid_y = train_comments['attack'], valid_comments['attack']

		# # label encode the target variable 
		# encoder = preprocessing.LabelEncoder()
		# train_y = encoder.fit_transform(train_y)
		# valid_y = encoder.fit_transform(valid_y)

		# create a count vectorizer object 
		count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		# word level tf-idf
		tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		# ngram level tf-idf 
		tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
		# characters level tf-idf
		tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

		clf = Pipeline([
		    ('vect', tfidf_vect),
		    ('clf', LogisticRegression()),
		])
		clf = clf.fit(train_comments['comment'], train_comments['attack'])
		auc = roc_auc_score(valid_comments['attack'], clf.predict_proba(valid_comments['comment'])[:, 1])
		#print('Test ROC AUC: %.3f' %auc)

		# load the pre-trained word-embedding vectors 
		# embeddings_index = {}
		# for i, line in enumerate(open('http://127.0.0.1:8000/static/wiki-news-300d-1M.vec', encoding="utf8")):
		# 	values = line.split()
		# 	embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

		# # create a tokenizer 
		# token = text.Tokenizer()
		# token.fit_on_texts(comments['comment'])
		# word_index = token.word_index

		# # convert text to sequence of tokens and pad them to ensure equal length vectors 
		# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
		# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

		# correctly classify nasty comment
		all_sen=""
		for sen in text:
		    x = clf.predict([sen])
		    if x==True:
		        all_sen=all_sen+sen+'.'


		# # correctly classify nasty comment
		# y=clf.predict(['People as stupid as you should not edit Wikipedia!'])
		# print(y)


		lexicon = Empath()
		x=lexicon.analyze(all_sen, categories=["negative_emotion", "positive_emotion"], normalize=True)

		if all_sen =='':
			all_sen="No Hate Sentence Found"
			x={'negative_emotion':0,"positive_emotion":0}

		appointments= {
		"query": x,
		"cl_name":": Linear Classifier on Word Level TF IDF Vectors",
		"text":all_sen,
		"auc":auc,
		"accuracy":"Classifier Accuracy :",
		"aggression":"Attack",
		"cattack":"Attack",
		"sentence":"Identified Attack Sentences :",
		"Liwc_pos":"Positve Emotion :",
		"Liwc_neg":"Negative Emotion :",
		
		}
		return render(request, 'index.html', appointments )
	else:
		appointments= {
		"cattack":"Attack",
	
		}
		return render(request, 'index.html', appointments )

def toxicity(request):
	
	if request.GET.get("q"):
		url = request.GET.get("q")
		html = urllib.request.urlopen(url).read().decode('utf-8')
		text = get_text(html)
		text=text.strip()
		text=text.split('.')

		comments = pd.read_csv('http://127.0.0.1:8000/static/toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations = pd.read_csv('http://127.0.0.1:8000/static/toxicity_annotations.tsv',  sep = '\t')
		len(annotations['rev_id'].unique())

		# labels a comment as an atack if the majority of annoatators did so
		labels = annotations.groupby('rev_id')['toxicity'].mean() > 0.5
		# print(labels)

		# join labels and comments
		comments['toxicity'] = labels

		# =========================== Data Cleaning =====================

		# clean the text
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.lower())
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('.,[^a-zA-z0-9\s]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(' +',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub(':',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('`',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('>',' ',x)))
		comments['comment'] = comments['comment'].apply((lambda x: re.sub('<',' ',x)))

		# remove newline and tab tokens
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

		hd=comments.query('toxicity')['comment'].head()

		# keeping only training and test sets
		train_comments = comments.query("split=='train'")
		valid_comments = comments.query("split=='test'")

		# split the dataset into training and validation datasets 
		train_x, valid_x = train_comments['comment'], valid_comments['comment'], 
		train_y, valid_y = train_comments['toxicity'], valid_comments['toxicity']

		# # label encode the target variable 
		# encoder = preprocessing.LabelEncoder()
		# train_y = encoder.fit_transform(train_y)
		# valid_y = encoder.fit_transform(valid_y)

		# create a count vectorizer object 
		count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		# word level tf-idf
		tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		# ngram level tf-idf 
		tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
		# characters level tf-idf
		tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

		clf = Pipeline([
		    ('vect', tfidf_vect),
		    ('clf', LogisticRegression()),
		])
		clf = clf.fit(train_comments['comment'], train_comments['toxicity'])
		auc = roc_auc_score(valid_comments['toxicity'], clf.predict_proba(valid_comments['comment'])[:, 1])
		#print('Test ROC AUC: %.3f' %auc)

		# load the pre-trained word-embedding vectors 
		# embeddings_index = {}
		# for i, line in enumerate(open('http://127.0.0.1:8000/static/wiki-news-300d-1M.vec', encoding="utf8")):
		# 	values = line.split()
		# 	embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

		# # create a tokenizer 
		# token = text.Tokenizer()
		# token.fit_on_texts(comments['comment'])
		# word_index = token.word_index

		# # convert text to sequence of tokens and pad them to ensure equal length vectors 
		# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
		# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

		# correctly classify nasty comment
		all_sen=""
		for sen in text:
		    x = clf.predict([sen])
		    if x==True:
		        all_sen=all_sen+sen+'.'


		# # correctly classify nasty comment
		# y=clf.predict(['People as stupid as you should not edit Wikipedia!'])
		# print(y)


		lexicon = Empath()
		x=lexicon.analyze(all_sen, categories=["negative_emotion", "positive_emotion"], normalize=True)

		if all_sen =='':
			all_sen="No Hate Sentence Found"
			x={'negative_emotion':0,"positive_emotion":0}

		appointments= {
		"query": x,
		"cl_name":": Linear Classifier on Word Level TF IDF Vectors",
		"text":all_sen,
		"auc":auc,
		"accuracy":"Classifier Accuracy :",
		"aggression":"Toxicity",
		"ctoxicity":"Toxicity",
		"sentence":"Identified Toxicity Sentences :",
		"Liwc_pos":"Positve Emotion :",
		"Liwc_neg":"Negative Emotion :",
		
		}
		return render(request, 'index.html', appointments )
	else:
		appointments= {
		"ctoxicity":"Toxicity",
	
		}
		return render(request, 'index.html', appointments )


def allhate(request):
	
	if request.GET.get("q"):
		url = request.GET.get("q")
		html = urllib.request.urlopen(url).read().decode('utf-8')
		text = get_text(html)
		text=text.strip()
		text=text.split('.')


		# Read data files
		comments_attack = pd.read_csv('http://127.0.0.1:8000/static/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations_attack = pd.read_csv('http://127.0.0.1:8000/static/attack_annotations.tsv',  sep = '\t')
		comments_aggression = pd.read_csv('http://127.0.0.1:8000/static/aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations_aggression = pd.read_csv('http://127.0.0.1:8000/static/aggression_annotations.tsv',  sep = '\t')
		comments_toxicity = pd.read_csv('http://127.0.0.1:8000/static/toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
		annotations_toxicity = pd.read_csv('http://127.0.0.1:8000/static/toxicity_annotations.tsv',  sep = '\t')
		
		# # labels a comment as an atack if the majority of annoatators did so
		# labels = annotations.groupby('rev_id')['attack'].mean() > 0.5


		# labels a comment if the majority of annoatators did so
		labels_attack = annotations_attack.groupby('rev_id')['attack'].mean() > 0.5
		labels_aggression = annotations_aggression.groupby('rev_id')['aggression'].mean() > 0.5
		labels_toxicity = annotations_toxicity.groupby('rev_id')['toxicity'].mean() > 0.5

		# join labels and comments
		comments_attack['label'] = labels_attack
		comments_aggression['label'] = labels_aggression
		comments_toxicity['label'] = labels_toxicity

		# Take only Attack, Aggression, Toxicity
		neutral01=comments_attack.query("label == False")
		neutral02=comments_aggression.query("label == False")
		neutral03=comments_toxicity.query("label == False")
		comments_attack = comments_attack.query("label == True")
		comments_aggression = comments_aggression.query("label == True")
		comments_toxicity = comments_toxicity.query("label == True")

		# labels: Neutral=0, Attack = 1, Aggression =2, Toxicity = 3
		neutral=pd.concat([neutral01, neutral02, neutral03], axis = 0)
		neutral['label']=0
		comments_attack['label'] = comments_attack['label']=1
		comments_aggression['label'] = comments_aggression['label']=2
		comments_toxicity['label'] = comments_toxicity['label']=3

		# Concatenation of the three data sets
		dataframe = pd.concat([neutral, comments_attack, comments_aggression, comments_toxicity], axis = 0)
		dataframe.iloc[20000]['comment']
		len(dataframe)

		# Text preprocessing
		dataframe['comment'] = dataframe['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
		dataframe['comment'] = dataframe['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
		dataframe['comment'] = dataframe['comment'].apply(lambda x: x.lower())

		dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
		dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub('[^0-9a-z #+_]',' ',x)))

		dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub(' +',' ',x)))




		# Take only Attack, Aggression, Toxicity
		neutral01=comments_attack.query("label == False")
		neutral02=comments_aggression.query("label == False")
		neutral03=comments_toxicity.query("label == False")
		comments_attack = comments_attack.query("label == True")
		comments_aggression = comments_aggression.query("label == True")
		comments_toxicity = comments_toxicity.query("label == True")


		# ngram level tf-idf 
		tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3,4), max_features=5000)
		tfidf_vect_ngram.fit(dataframe['comment'])

		#split the data into training and validation sets
		train_x, valid_x, train_y, valid_y = train_test_split(dataframe['comment'], dataframe['label'], test_size=0.2, random_state=42)


		clf = Pipeline([
		    ('vect', tfidf_vect_ngram),
		    ('clf', LogisticRegression()),
		])

		clf = clf.fit(train_x, train_y)
		predictions = clf.predict(valid_x)
		auc= metrics.accuracy_score(predictions, valid_y)


		#print('Test ROC AUC: %.3f' %auc)

		# load the pre-trained word-embedding vectors 
		# embeddings_index = {}
		# for i, line in enumerate(open('http://127.0.0.1:8000/static/wiki-news-300d-1M.vec', encoding="utf8")):
		# 	values = line.split()
		# 	embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

		# # create a tokenizer 
		# token = text.Tokenizer()
		# token.fit_on_texts(comments['comment'])
		# word_index = token.word_index

		# # convert text to sequence of tokens and pad them to ensure equal length vectors 
		# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
		# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

		# # correctly classify nasty comment
		# all_sen=""
		# for sen in text:
		#     x = clf.predict([sen])
		#     if x==True:
		#         all_sen=all_sen+sen

		all_sen=""
		for sen in text:
			x = clf.predict([sen])
			if x==1:
				all_sen=all_sen+sen+'.'
			elif x==2:
				all_sen=all_sen+sen+'.'
			elif x==3:
				all_sen=all_sen+sen+'.'


		# # correctly classify nasty comment
		# y=clf.predict(['People as stupid as you should not edit Wikipedia!'])
		# print(y)


		lexicon = Empath()
		x=lexicon.analyze(all_sen, categories=["negative_emotion", "positive_emotion"], normalize=True)

		if all_sen =='':
			all_sen="No Hate Sentence Found"
			x={'negative_emotion':0,"positive_emotion":0}

		appointments= {
		"query": x,
		"cl_name":": Linear Classifier on Word Level TF IDF Vectors",
		"text":all_sen,
		"auc":auc,
		"accuracy":"Classifier Accuracy :",
		"aggression":"All Hate Speech",
		"call":"All",
		"sentence":"Identified Hate Sentences :",
		"Liwc_pos":"Positve Emotion :",
		"Liwc_neg":"Negative Emotion :",
		
		}
		return render(request, 'index.html', appointments )
	else:
		appointments= {
		"call":"All",
	
		}
		return render(request, 'index.html', appointments )
