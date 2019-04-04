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

def get_labels(labels_filename):
	labels_holder=np.genfromtxt(labels_filename,dtype='str',delimiter='\n')
	labels_indicies=[]
	labels=[]
	for index in range(0,len(labels_holder),1):
		l=labels_holder[index].split(" ")
		if(l[0]!="EMPTY"):
			labels_indicies.append(index)
			labels.append(l[0])
	return labels,labels_indicies

def get_data(data_filename,indicies,train_indicies,test_indicies):
	data_holder=np.genfromtxt(data_filename,dtype='str',delimiter='\n')
	data=[]
	for index in indicies:
		data.append(data_holder[index])
	
	train_data=[]
	test_data=[]
	for index in train_indicies:
		train_data.append(data[index])
	
	for index in test_indicies:
		test_data.append(data[index])
		
	return train_data,test_data

def random_sample(number,labels):
	idx = np.random.randint(number, size=number)
	train_data=[]
	test_data=[]
	train_data_indicies=[]
	test_data_indicies=[]
	for i in idx:
		train_data.append(labels[i])
		train_data_indicies.append(i)
	
	for i in range(0,len(labels),1):
		if(i not in idx):
			test_data.append(labels[i])
			test_data_indicies.append(i)
	return train_data,train_data_indicies,test_data,test_data_indicies

def dot_sum_of_prob_feature_vector(weights,people,orgs,dates,exchanges):
	weighted_people=np.multiply(weights[0],people)
	weighted_orgs=np.multiply(weights[1],orgs)
	weighted_dates=np.multiply(weights[2],dates)
	weighted_exchanges=np.multiply(weights[3],exchanges)
	together_1=np.add(weighted_people,weighted_orgs)
	together_2=np.add(weighted_dates,weighted_exchanges)
	together=np.add(together_1,together_2)
	return together
	
def dot_sum_of_feature_vectors(weights,unique,body,combined):
	weighted_unique=np.multiply(weights[0],unique)
	weighted_body=np.multiply(weights[1],body)
	weighted_combined=np.multiply(weights[2],combined)
	together_1=np.add(weighted_unique,weighted_body)
	together=np.add(together_1,weighted_combined)
	return together

def classifyTitlesTopics(topics_train_data,topics_train_labels,topics_test_data,topics_test_labels):
	topics_vectorizer = CountVectorizer()
	topics_X = topics_vectorizer.fit_transform(topics_train_data)
	topics_Y= topics_vectorizer.transform(topics_test_data)

	neigh = KNeighborsClassifier(n_neighbors=6)
	neigh.fit(topics_X, topics_train_labels)
	
	topics_prob_labels=neigh.predict_proba(topics_Y)
	
	return topics_prob_labels,neigh.classes_

def classifyDatelinesPlaces(places_train_data,places_train_labels,places_test_data,places_test_labels):
	places_vectorizer = CountVectorizer()
	places_X = places_vectorizer.fit_transform(places_train_data)
	places_Y= places_vectorizer.transform(places_test_data)

	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(places_X, places_train_labels)
	
	places_predicted_labels=neigh.predict(places_Y)
	places_prob_labels=neigh.predict_proba(places_Y)
	return places_prob_labels,neigh.classes_
   
def classifyTopicsPlaces(train_data,train_labels,test_data,test_labels,k):
   vectorizer = CountVectorizer()
   X = vectorizer.fit_transform(train_data)
   Y= vectorizer.transform(test_data)

   neigh = KNeighborsClassifier(n_neighbors=k)
   neigh.fit(X, train_labels)
   predicted_prob=neigh.predict_proba(Y)
   return predicted_prob

def test(probs,labels,test_labels):
	predicted_labels=[]
	for prob in probs:
		maxIndex=prob.tolist().index(max(prob))
		predicted_labels.append(labels[maxIndex])
		
	classificationReport = classification_report(predicted_labels, test_labels)
	print("Classification report for testing set: ")
	print(classificationReport)
	print("Accuracy score for testing set: ")
	print(accuracy_score(predicted_labels, test_labels))
	
def main():
	print('loading labels')
	#load labels, get indicies
	#load places array
	places_labels,places_indicies=get_labels('reduced_data/places_labels.out')
	#get random sample, return indicies
	places_train_labels,places_train_indicies,places_test_labels,places_test_indicies=random_sample(int(np.floor(0.80*len(places_labels))),places_labels)
	#load topics array
	topics_labels,topics_indicies=get_labels('reduced_data/topics_labels.out')
	#get random smample, return indicies
	topics_train_labels,topics_train_indicies,topics_test_labels,topics_test_indicies=random_sample(int(np.floor(0.80*len(topics_labels))),topics_labels)
	
	print('loading places data')
	#load places data
	train_datelines,test_datelines=get_data('reduced_data/reduced_datelines.out',places_indicies,places_train_indicies,places_test_indicies)
	train_places_body,test_places_body=get_data('reduced_data/reduced_bodies.out',places_indicies,places_train_indicies,places_test_indicies)
	train_places_people,test_places_people=get_data('reduced_data/reduced_people.out',places_indicies,places_train_indicies,places_test_indicies)
	train_places_orgs,test_places_orgs=get_data('reduced_data/reduced_orgs.out',places_indicies,places_train_indicies,places_test_indicies)
	train_places_dates,test_places_dates=get_data('reduced_data/reduced_dates.out',places_indicies,places_train_indicies,places_test_indicies)
	train_places_exchanges,test_places_exchanges=get_data('reduced_data/reduced_exchanges.out',places_indicies,places_train_indicies,places_test_indicies)
	
	print('training for places')
	#start training places
	
	print('training places and datelines')
	places_datelines_prob,datelines_labels=classifyDatelinesPlaces(train_datelines,places_train_labels,test_datelines,places_test_labels)
	
	print('training places and bodies')
	places_bodies_prob=classifyTopicsPlaces(train_places_body,places_train_labels,test_places_body,places_test_labels,6)
	
	print('training places and people')
	places_people_prob=classifyTopicsPlaces(train_places_people,places_train_labels,test_places_people,places_test_labels,6)
	
	print('training places and orgs')
	places_orgs_prob=classifyTopicsPlaces(train_places_orgs,places_train_labels,test_places_orgs,places_test_labels,6)
	
	print('training places and dates')
	places_dates_prob=classifyTopicsPlaces(train_places_dates,places_train_labels,test_places_dates,places_test_labels,6)
	
	print('training places and exchanges')
	places_exchanges_prob=classifyTopicsPlaces(train_places_exchanges,places_train_labels,test_places_exchanges,places_test_labels,6)
	
	
	sorted_places_labels=sorted(datelines_labels)
	
	combined_places_weights=[0.4,0.1,0.2,0.35]
	combined_predicted_probs=dot_sum_of_prob_feature_vector(combined_places_weights,places_people_prob,places_orgs_prob,places_dates_prob,places_exchanges_prob)
	vectors_places_weights=[0.7,0.2,0.1]
	places_predicted_probs=dot_sum_of_feature_vectors(vectors_places_weights,places_datelines_prob,places_bodies_prob,combined_predicted_probs)
	print('testing places')
	test(places_predicted_probs,sorted_places_labels,places_test_labels)
	
	print('loading topics data')
	#load topics data
	train_titles,test_titles=get_data('reduced_data/reduced_titles.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	train_topics_body,test_topics_body=get_data('reduced_data/reduced_bodies.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	train_topics_people,test_topics_people=get_data('reduced_data/reduced_people.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	train_topics_orgs,test_topics_orgs=get_data('reduced_data/reduced_orgs.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	train_topics_dates,test_topics_dates=get_data('reduced_data/reduced_dates.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	train_topics_exchanges,test_topics_exchanges=get_data('reduced_data/reduced_exchanges.out',topics_indicies,topics_train_indicies,topics_test_indicies)
	
	print('training for topics')
	print('training topics and datelines')
	topics_titles_prob,titles_labels=classifyTitlesTopics(train_titles,topics_train_labels,test_titles,topics_test_labels)
	
	print('training topics and bodies')
	topics_bodies_prob=classifyTopicsPlaces(train_topics_body,topics_train_labels,test_topics_body,topics_test_labels,6)
	
	print('training topics and people')
	topics_people_prob=classifyTopicsPlaces(train_topics_people,topics_train_labels,test_topics_people,topics_test_labels,6)
	
	print('training topics and orgs')
	topics_orgs_prob=classifyTopicsPlaces(train_topics_orgs,topics_train_labels,test_topics_orgs,topics_test_labels,6)
	
	print('training topics and dates')
	topics_dates_prob=classifyTopicsPlaces(train_topics_dates,topics_train_labels,test_topics_dates,topics_test_labels,6)
	
	print('training topics and exchanges')
	topics_exchanges_prob=classifyTopicsPlaces(train_topics_exchanges,topics_train_labels,test_topics_exchanges,topics_test_labels,6)
	
	
	sorted_topics_labels=sorted(titles_labels)
	
	combined_topics_weights=[0.7,0.08,0.08,0.14]
	combined_predicted_probs=dot_sum_of_prob_feature_vector(combined_topics_weights,topics_people_prob,topics_orgs_prob,topics_dates_prob,topics_exchanges_prob)
	vectors_topics_weights=[0.2,0.4,0.4]
	topics_predicted_probs=dot_sum_of_feature_vectors(vectors_topics_weights,topics_titles_prob,topics_bodies_prob,combined_predicted_probs)
	
	print('testing topics')
	test(topics_predicted_probs,sorted_topics_labels,topics_test_labels)
	

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)