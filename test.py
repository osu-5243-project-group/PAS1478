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

def main():
	data=np.genfromtxt('reduced_titles_array.out',dtype='str',delimiter='\n')
	print(len(data))
	data_labels=np.genfromtxt('topics_labels.out',dtype='str',delimiter='\n')
	print(len(data_labels))
	
	labels=[]
	for label in data_labels:
		l=label.split(" ")
		if(len(l)<=0):
			labels.append("EMPTY")
		else:
			labels.append(l[0])
	
	train_data=data[0:20000]
	test_data=data[20001:20500]
	train_labels=labels[0:20000]
	test_labels=labels[20001:20500]
	
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(train_data)
	Y= vectorizer.transform(test_data)
	#print(vectorizer.get_feature_names())
	#print(X.toarray())
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X, train_labels)
	
	predicted_labels=neigh.predict(Y)

	classificationReport = classification_report(test_data, test_labels)
	print("Classification report for testing set: ")
	print(classificationReport)
	
	print("Accuracy score for testing set: ")
	print(accuracy_score(predicted_labels, test_labels))
	

main()