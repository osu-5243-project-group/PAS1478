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
	data,labels=get_data('reduced_data/reduced_titles_array.out','reduced_data/topics_labels.out')
	print(len(data))
	print(len(labels))
	train_data=data[0:11000]
	test_data=data[11001:11366]
	train_labels=labels[0:11000]
	test_labels=labels[11001:11366]
	
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(train_data)
	Y= vectorizer.transform(test_data)
	#print(vectorizer.get_feature_names())
	#print(X.toarray())
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X, train_labels)
	
	predicted_labels=neigh.predict(Y)

	print(predicted_labels)
	classificationReport = classification_report(predicted_labels, test_labels)
	print("Classification report for testing set: ")
	print(classificationReport)
	
	print("Accuracy score for testing set: ")
	print(accuracy_score(predicted_labels, test_labels))
	

main()