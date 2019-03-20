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
	filenames=glob('*.sgm')
	return filenames

def get_datelines(soup,datelines_array):
	objects=soup.find_all('dateline')
	for object in objects:
		months=['January','February','March','April','May','June','July','August','September','October','November','December','Jan','Feb','Mar','Apr','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
		string=object.string.lower()
		for month in months:
			check=object.string.lower().find(month.lower())
			if(check!=-1):
				string=object.string.lower()[0:check-2]
		if(string not in datelines_array):
			datelines_array.append(string.lower())
	return datelines_array

def get_objects(soup,type,array,empty_count,counts):
	objects=soup.find_all(type)
	for object in objects:
		empty_check=True
		ds=object.find_all('d')
		for d in ds:
			empty_check=False
			if(d.string in counts):
				counts[d.string]=counts[d.string]+1
			else:
				counts[d.string]=1
			array.append(d.string)
		if(empty_check):
			empty_count=empty_count+1
	return array,empty_count,counts

def get_unique(data):
	data_set = set(data)
	return data_set
	
def frequency_of_words_in_body(soup,counts):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	objects=soup.find_all('body')
	porter=PorterStemmer()
	for object in objects:
		tokens=word_tokenize(object.string)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				print(stemmed)
				if(stemmed in counts):
					counts[stemmed]=counts[stemmed]+1
				else:
					counts[stemmed]=1
	return counts
	
def write_words(soup,array):
	stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
	objects=soup.find_all('title')
	porter=PorterStemmer()
	for object in objects:
		join_string=" "
		join_array=[]
		tokens=word_tokenize(object.string)
		#words=object.string.split("[- :,] ",",","!",".","\s")
		for word in tokens:
			if(word not in stop_words and word.isalpha()):
				stemmed=porter.stem(word)
				join_array.append(stemmed)
		join_string=join_string.join(join_array)
		array.append(join_string)	
	return array

def main():
	places_array=[]
	places_counts={}
	topics_array=[]
	topics_counts={}
	datelines_array=[]
	places_empty=0
	topics_empty=0
	filenames=get_filenames()
	word_counts={}
	array=[]
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		array=write_words(soup,array)
		#frequency_of_words_in_body(soup,word_counts)
		#datelines_array=get_datelines(soup,datelines_array)
		#places_array,places_empty,places_counts=get_objects(soup,'places',places_array,places_empty,places_counts)
		#topics_array,topics_empty,topics_counts=get_objects(soup,'topics',topics_array,topics_empty,topics_counts)
	np.savetxt('reduced_array.out', array, delimiter='\n', fmt='%s')
	
	# unique_places=get_unique(places_array)
	# unique_topics=get_unique(topics_array)
	# print('list of unique places: ', unique_places)
	# print('list of unique topics: ',unique_topics)
	# print('number of unique places: ', len(unique_places))
	# print('number of unique topics: ', len(unique_topics))
	# print('number of empty places objects: ',places_empty)
	# print('number of empty topics objects: ',topics_empty)
	
	print('unique datelines (minus dates):', datelines_array)
	print('number of unique datelines:', len(datelines_array))
	
main()