import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import sys
	
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

def classifyTopicsPlaces(train_data,train_labels,test_data,test_labels,k):
   vectorizer = CountVectorizer()
   X = vectorizer.fit_transform(train_data)
   Y= vectorizer.transform(test_data)

   neigh = KNeighborsClassifier(n_neighbors=k)
   neigh.fit(X, train_labels)
   predicted_labels=neigh.predict(Y)
   print("Accuracy score for testing set: ")
   print(accuracy_score(predicted_labels, test_labels))

def get_accuracies(type,train_data,train_labels,test_data,test_labels):
	K=[1,2,3,4,5,6,7,8,9,10]
	accuracies=[]
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(train_data)
	Y= vectorizer.transform(test_data)
	for k in K:
		neigh = KNeighborsClassifier(n_neighbors=k)
		neigh.fit(X, train_labels)
		predicted_labels=neigh.predict(Y)
		accuracies.append(accuracy_score(predicted_labels, test_labels))
	plt.plot(K,accuracies, '-', color='red')
	title=type+' Accuracies vs K'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('Accuracies')
	plt.show()
	return accuracies
	
def plot(type,accuracies):
	if(type=='places'):
		plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['datelines'], '-', color='blue', label='datelines')
	else:
		plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['titles'], '-', color='blue', label='titles')
	plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['bodies'], '-', color='red', label='bodies')
	plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['orgs'], '-', color='yellow', label='orgs')
	plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['people'], '-', color='green', label='people')
	plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['exchanges'], '-', color='magenta', label='exchanges')
	plt.plot([1,2,3,4,5,6,7,8,9,10],accuracies['dates'], '-', color='cyan', label='dates')
	title=type+ ' Accuracies vs K'
	plt.title(title)
	plt.xlabel('K')
	plt.ylabel('Accuracies')
	plt.legend(loc='center right')
	plt.show()

def main():
	places_accuracies={}
	
	print('places and datelines')
	places_data,places_labels=get_data('reduced_data/reduced_datelines.out','reduced_data/places_labels.out')
	places_train_data,places_train_labels,places_test_data,places_test_labels=random_sample(int(np.floor(0.80*len(places_data))),places_data,places_labels)
	# classifyTopicsPlaces(places_train_data,places_train_labels,places_test_data,places_test_labels,6)
	places_accuracies['datelines']=get_accuracies('Datelines Places',places_train_data,places_train_labels,places_test_data,places_test_labels)

	print('places and bodies')
	bodies_data,bodies_labels=get_data('reduced_data/reduced_bodies.out','reduced_data/places_labels.out')
	bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels=random_sample(int(np.floor(0.80*len(bodies_data))),bodies_data,bodies_labels)
	# classifyTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels,6)
	places_accuracies['bodies']=get_accuracies('Bodies Places',bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)

	print('places and orgs')
	orgs_data,orgs_labels=get_data('reduced_data/reduced_orgs.out','reduced_data/topics_labels.out')
	orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels=random_sample(int(np.floor(0.80*len(orgs_data))),orgs_data,orgs_labels)
	# classifyTopicsPlaces(orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels,6)
	places_accuracies['orgs']=get_accuracies('Orgs Places',orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels)

	print('places and exchanges')
	exchanges_data,exchanges_labels=get_data('reduced_data/reduced_exchanges.out','reduced_data/places_labels.out')
	exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels=random_sample(int(np.floor(0.80*len(exchanges_data))),exchanges_data,exchanges_labels)
	# classifyTopicsPlaces(exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels,6)
	places_accuracies['exchanges']=get_accuracies('Exchanges Places',exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels)

	print('places and dates')
	dates_data,dates_labels=get_data('reduced_data/reduced_dates.out','reduced_data/places_labels.out')
	dates_train_data,dates_train_labels,dates_test_data,dates_test_labels=random_sample(int(np.floor(0.80*len(dates_data))),dates_data,dates_labels)
	# classifyTopicsPlaces(dates_train_data,dates_train_labels,dates_test_data,dates_test_labels,6)
	places_accuracies['dates']=get_accuracies('Dates Places',dates_train_data,dates_train_labels,dates_test_data,dates_test_labels)

	print('places and people')
	people_data,people_labels=get_data('reduced_data/reduced_people.out','reduced_data/places_labels.out')
	people_train_data,people_train_labels,people_test_data,people_test_labels=random_sample(int(np.floor(0.80*len(people_data))),people_data,bodies_labels)
	# classifyTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels,6)
	places_accuracies['people']=get_accuracies('People Places',people_train_data,people_train_labels,people_test_data,people_test_labels)

	plot('places',places_accuracies)

	topics_accuracies={}
	print('topics and titles')
	topic_data,topic_labels=get_data('reduced_data/reduced_titles.out','reduced_data/topics_labels.out')
	topic_train_data,topic_train_labels,topic_test_data,topic_test_labels=random_sample(int(np.floor(0.80*len(topic_data))),topic_data,topic_labels)
	# classifyTopicsPlaces(topic_train_data,topic_train_labels,topic_test_data,topic_test_labels,6)
	topics_accuracies['titles']=get_accuracies('Titles Topics',topic_train_data,topic_train_labels,topic_test_data,topic_test_labels)

	print('topics and bodies')
	bodies_data,bodies_labels=get_data('reduced_data/reduced_bodies.out','reduced_data/topics_labels.out')
	bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels=random_sample(int(np.floor(0.80*len(bodies_data))),bodies_data,bodies_labels)
	# classifyTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels,6)
	topics_accuracies['bodies']=get_accuracies('Bodies Topics',bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)
	
	print('topics and orgs')
	orgs_data,orgs_labels=get_data('reduced_data/reduced_orgs.out','reduced_data/topics_labels.out')
	orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels=random_sample(int(np.floor(0.80*len(orgs_data))),orgs_data,orgs_labels)
	# classifyTopicsPlaces(orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels,6)
	topics_accuracies['orgs']=get_accuracies('Orgs Topics',orgs_train_data,orgs_train_labels,orgs_test_data,orgs_test_labels)
	
	print('topics and exchanges')
	exchanges_data,exchanges_labels=get_data('reduced_data/reduced_exchanges.out','reduced_data/topics_labels.out')
	exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels=random_sample(int(np.floor(0.80*len(exchanges_data))),exchanges_data,exchanges_labels)
	# classifyTopicsPlaces(exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels,6)
	topics_accuracies['exchanges']=get_accuracies('Exchanges Topics',exchanges_train_data,exchanges_train_labels,exchanges_test_data,exchanges_test_labels)
	
	print('topics and dates')
	dates_data,dates_labels=get_data('reduced_data/reduced_dates.out','reduced_data/topics_labels.out')
	dates_train_data,dates_train_labels,dates_test_data,dates_test_labels=random_sample(int(np.floor(0.80*len(dates_data))),dates_data,dates_labels)
	# classifyTopicsPlaces(dates_train_data,dates_train_labels,dates_test_data,dates_test_labels,6)
	topics_accuracies['dates']=get_accuracies('Dates Topics',dates_train_data,dates_train_labels,dates_test_data,dates_test_labels)
	
	print('topics and people')
	people_data,people_labels=get_data('reduced_data/reduced_people.out','reduced_data/topics_labels.out')
	people_train_data,people_train_labels,people_test_data,people_test_labels=random_sample(int(np.floor(0.80*len(people_data))),people_data,bodies_labels)
	# classifyTopicsPlaces(bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels,6)
	topics_accuracies['people']=get_accuracies('People Topics',bodies_train_data,bodies_train_labels,bodies_test_data,bodies_test_labels)
	
	plot('Topics',topics_accuracies)
	
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)
		
#main()