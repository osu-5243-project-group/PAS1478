from bs4 import BeautifulSoup
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

def get_data(data_filename,labels_filename):
	data_holder=np.genfromtxt(data_filename,dtype='str',delimiter='\n')
	labels_holder=np.genfromtxt(labels_filename,dtype='str',delimiter='\n')
	labels=[]
	data=[]
	
	for index in range(0,len(labels_holder),1):
		l=labels_holder[index].split(" ")
		if(l[0]!="EMPTY"):
			data.append(data_holder[index])
			labels.append(l[0])
	return data,labels

def random_sample(number,data,labels):
	idx = np.random.randint(number, size=number)
	data_train=[]
	data_train_labels=[]
	data_test=[]
	data_test_labels=[]
	for i in idx:
		data_train.append(data[i])
		data_train_labels.append(labels[i])
	
	for i in range(0,len(data),1):
		check=True
		if(i not in idx):
			data_test.append(data[i])
			data_test_labels.append(labels[i])
			
	return data_train,data_train_labels,data_test,data_test_labels

def classifyTitlesTopics(topic_train_data,topic_train_labels,topic_test_data,topic_test_labels):
	topic_vectorizer = CountVectorizer()
	topic_X = topic_vectorizer.fit_transform(topic_train_data)
	topic_Y= topic_vectorizer.transform(topic_test_data)

	neigh = KNeighborsClassifier(n_neighbors=6)
	neigh.fit(topic_X, topic_train_labels)
	
	#topic_predicted_labels=neigh.predict(topic_Y)
	topic_prob_labels=neigh.predict_proba(topic_Y)
	# print(topic_prob_labels)
	# print(topic_predicted_labels)
	# topic_classificationReport = classification_report(topic_predicted_labels, topic_test_labels)
	# print("Classification report for testing set: ")
	# print(topic_classificationReport)
	
	# print("Accuracy score for testing set: ")
	# print(accuracy_score(topic_predicted_labels, topic_test_labels))
	
	return topic_prob_labels

def classifyDatelinesPlaces(places_train_data,places_train_labels,places_test_data,places_test_labels):
	places_vectorizer = CountVectorizer()
	places_X = places_vectorizer.fit_transform(places_train_data)
	places_Y= places_vectorizer.transform(places_test_data)

	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(places_X, places_train_labels)
	
	#places_predicted_labels=neigh.predict(places_Y)
	places_prob_labels=neigh.predict_proba(places_Y)
	# places_classificationReport = classification_report(places_predicted_labels, places_test_labels)
	# print("Classification report for testing set: ")
	# print(places_classificationReport)
	
	# print("Accuracy score for testing set: ")
	# print(accuracy_score(places_predicted_labels, places_test_labels))
	return places_prob_labels
	
def classifyBodiesTopicsPlaces(train_data,train_labels,test_data,test_labels):
   vectorizer = CountVectorizer()
   X = vectorizer.fit_transform(train_data)
   Y= vectorizer.transform(test_data)

   neigh = KNeighborsClassifier(n_neighbors=6)
   neigh.fit(X, train_labels)

   predicted_labels=neigh.predict(Y)

   print(predicted_labels)
   topic_classificationReport = classification_report(predicted_labels, test_labels)
   print("Classification report for testing set: ")
   print(classificationReport)

   print("Accuracy score for testing set: ")
   print(accuracy_score(predicted_labels, test_labels))

def main():      
	print('topics and titles') #0.58 accuracy
	topic_data,topic_labels=get_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
	topic_train_data,topic_train_labels,topic_test_data,topic_test_labels=random_sample(int(np.floor(0.80*len(topic_data))),topic_data,topic_labels)
	topic_titles_prob=classifyTitlesTopics(topic_train_data,topic_train_labels,topic_test_data,topic_test_labels)

	print('places and datelines') #0.93 accuracy
	places_data,places_labels=get_data('reduced_data/reduced_datelines.out','reduced_data/places_labels.out')
	places_train_data,places_train_labels,places_test_data,places_test_labels=random_sample(int(np.floor(0.80*len(places_data))),places_data,places_labels)
	places_datelines_prob=classifyDatelinesPlaces(places_train_data,places_train_labels,places_test_data,places_test_labels)
	
	print('topics and bodies')
	bodies_data,bodies_labels=get_data('reduced_data/reduced_bodies.out','reduced_data/places_labels.out')
	bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels=random_sample(int(np.floor(0.80*len(bodies_data))),bodies_data,bodies_labels)
	classifyBodiesTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)
	
	print('places and bodies')
	bodies_data,bodies_labels=get_data('reduced_data/reduced_bodies.out','reduced_data/topics_labels.out')
	bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels=random_sample(int(np.floor(0.80*len(bodies_data))),bodies_data,bodies_labels)
	classifyDatelinesPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)
	classifyBodiesTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)
	

main()