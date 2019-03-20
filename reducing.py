import nltk
#nltk.download('popular')
from bs4 import BeautifulSoup
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

def get_filenames():
	filenames=glob('data/*.sgm')
	return filenames

def write_labels(soup,array,type):
	objects=soup.find_all(type)
	for object in objects:
		temp=[]
		empty_check=True
		ds=object.find_all('d')
		for d in ds:
			temp.append(d.string.lower())
		join_string=" "
		join_string=join_string.join(temp)
		if(join_string==""):
			join_string="EMPTY"
		array.append(join_string)
	return array

def write_datelines(soup,array):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	articles=soup.find_all('reuters')
	porter=PorterStemmer()
	for article in articles:
		#print(article.title)
		#object=article_soup.find(type)
		if(article.dateline is not None):
			object=article.dateline.string
		else:
			object=""
		join_string=" "
		join_array=[]
		if(object is not None):
			tokens=word_tokenize(object)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				join_array.append(stemmed)
		join_string=join_string.join(join_array)
		if(join_string==""):
			join_string="EMPTY"
		array.append(join_string)
	return array

def write_words(soup,array,type):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	articles=soup.find_all('reuters')
	porter=PorterStemmer()
	for article in articles:
		#print(article.title)
		#object=article_soup.find(type)
		if(article.title is not None):
			object=article.title.string
		else:
			object=""
		join_string=" "
		join_array=[]
		if(object is not None):
			tokens=word_tokenize(object)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				join_array.append(stemmed)
		join_string=join_string.join(join_array)
		if(join_string==""):
			join_string="EMPTY"
		array.append(join_string)
	return array

def write_body(soup,array,type):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	articles=soup.find_all('reuters')
	porter=PorterStemmer()
	for article in articles:
		#print(article.title)
		#object=article_soup.find(type)
		if(article.body is not None):
			object=article.body.string
		else:
			object=""
		join_string=" "
		join_array=[]
		if(object is not None):
			tokens=word_tokenize(object)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				join_array.append(stemmed)
		join_string=join_string.join(join_array)
		if(join_string==""):
			join_string="EMPTY"
		array.append(join_string)
	return array

def write_org(soup,array,type):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	articles=soup.find_all('reuters')
	porter=PorterStemmer()
	for article in articles:
		#print(article.title)
		#object=article_soup.find(type)
		if(article.org is not None):
			object=article.org.string
		else:
			object=""
		join_string=" "
		join_array=[]
		if(object is not None):
			tokens=word_tokenize(object)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				join_array.append(stemmed)
		join_string=join_string.join(join_array)
		if(join_string==""):
			join_string="EMPTY"
		array.append(join_string)
	return array

def main():
	filenames=get_filenames()
	datelines=[]
	topics=[]
	places=[]
	titles=[]
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		datelines=write_datelines(soup,datelines)
		topics=write_labels(soup,topics,'topics')
		places=write_labels(soup,places,'places')
		titles=write_words(soup,titles,'titles')
	
	np.savetxt('reduced_titles.out', titles, delimiter='\n', fmt='%s')
	np.savetxt('places_labels.out', places, delimiter='\n', fmt='%s')
	np.savetxt('topics_labels.out', topics, delimiter='\n', fmt='%s')
	np.savetxt('reduced_datelines.out', datelines, delimiter='\n', fmt='%s')
	
main()