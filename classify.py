
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

def main():
	topic_data,topic_labels=get_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
	topic_train_data=topic_data[0:11000]
	topic_test_data=topic_data[11001:11366]
	topic_train_labels=topic_labels[0:11000]
	topic_test_labels=topic_labels[11001:11366]

	topic_vectorizer = CountVectorizer()
	topic_X = topic_vectorizer.fit_transform(topic_train_data)
	topic_Y= topic_vectorizer.transform(topic_test_data)

	neigh = KNeighborsClassifier(n_neighbors=6)
	neigh.fit(topic_X, topic_train_labels)

	topic_predicted_labels=neigh.predict(topic_Y)

	print(topic_predicted_labels)
	topic_classificationReport = classification_report(topic_predicted_labels, topic_test_labels)
	print("Classification report for testing set: ")
	print(topic_classificationReport)

	print("Accuracy score for testing set: ")
	print(accuracy_score(topic_predicted_labels, topic_test_labels))

	places_data,places_labels=get_data('reduced_data/reduced_datelines.out','reduced_data/places_labels.out')
	places_train_data=places_data[0:11000]
	places_test_data=places_data[11001:11366]
	places_train_labels=places_labels[0:11000]
	places_test_labels=places_labels[11001:11366]

	places_vectorizer = CountVectorizer()
	places_X = topic_vectorizer.fit_transform(places_train_data)
	places_Y= topic_vectorizer.transform(places_test_data)

	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(places_X, places_train_labels)

	places_predicted_labels=neigh.predict(places_Y)

	places_classificationReport = classification_report(places_predicted_labels, places_test_labels)
	print("Classification report for testing set: ")
	print(places_classificationReport)

	print("Accuracy score for testing set: ")
	print(accuracy_score(places_predicted_labels, places_test_labels))


main()
