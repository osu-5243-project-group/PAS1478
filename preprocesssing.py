from bs4 import BeautifulSoup
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
def get_filenames():
	filenames=glob('*.sgm')
	return filenames

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

def main():
	places_array=[]
	places_counts={}
	topics_array=[]
	topics_counts={}
	places_empty=0
	topics_empty=0
	filenames=get_filenames()
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		places_array,places_empty,places_counts=get_objects(soup,'places',places_array,places_empty,places_counts)
		topics_array,topics_empty,topics_counts=get_objects(soup,'topics',topics_array,topics_empty,topics_counts)
	
	print(places_counts)
	print(topics_counts)
	unique_places=get_unique(places_array)
	unique_topics=get_unique(topics_array)
	print('list of unique places: ', unique_places)
	print('list of unique topics: ',unique_topics)
	print('number of unique places: ', len(unique_places))
	print('number of unique topics: ', len(unique_topics))
	print('number of empty places objects: ',places_empty)
	print('number of empty topics objects: ',topics_empty)
	
main()