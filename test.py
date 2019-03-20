from bs4 import BeautifulSoup
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

def main():
	data=np.loadtxt('reduced_titles_array_out.out',dtype='str',delimiter='\n')
	data_labels=np.loadtxt('topics_labels.out',dtype='str',delimiter='\n')
	labels=[]
	for label in data_labels:
		l=label.split()
		labels.append(l)
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(data)
	print(vectorizer.get_feature_names())
	print(X.toarray())
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X, labels)

main()