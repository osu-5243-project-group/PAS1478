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
			print(object)
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
		array.append(join_string)
	return array

def main():
	filenames=get_filenames()
	array=[]
	topics=[]
	places=[]
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		array=write_words(soup,array,'title')
		#topics=write_labels(soup,topics,'topics')
		#places=write_labels(soup,places,'places')
		#frequency_of_words_in_body(soup,word_counts)
		#datelines_array=get_datelines(soup,datelines_array)
		#places_array,places_empty,places_counts=get_objects(soup,'places',places_array,places_empty,places_counts)
		#topics_array,topics_empty,topics_counts=get_objects(soup,'topics',topics_array,topics_empty,topics_counts)
	
	np.savetxt('reduced_titles_array.out', array, delimiter='\n', fmt='%s')
	#np.savetxt('places_labels.out', places, delimiter='\n', fmt='%s')
	#np.savetxt('topics_labels.out', topics, delimiter='\n', fmt='%s')
	
	# unique_places=get_unique(places_array)
	# unique_topics=get_unique(topics_array)
	# print('list of unique places: ', unique_places)
	# print('list of unique topics: ',unique_topics)
	# print('number of unique places: ', len(unique_places))
	# print('number of unique topics: ', len(unique_topics))
	# print('number of empty places objects: ',places_empty)
	# print('number of empty topics objects: ',topics_empty)
	
	#print('unique datelines (minus dates):', datelines_array)
	#print('number of unique datelines:', len(datelines_array))
	
main()