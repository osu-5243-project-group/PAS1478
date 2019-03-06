from bs4 import BeautifulSoup
from glob import glob

def get_filenames():
	filenames=glob('*.sgm')
	return filenames

def get_objects(soup,type,array,empty_count):
	objects=soup.find_all(type)
	for object in objects:
		empty_check=True
		ds=object.find_all('d')
		for d in ds:
			empty_check=False
			array.append(d.string)
		if(empty_check):
			empty_count=empty_count+1
	return array,empty_count

def get_unique(data):
	data_set = set(data)
	return data_set

def main():
	places_array=[]
	topics_array=[]
	places_empty=0
	topics_empty=0
	filenames=get_filenames()
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		places_array,places_empty=get_objects(soup,'places',places_array,places_empty)
		topics_array,topics_empty=get_objects(soup,'topics',topics_array,topics_empty)
	unique_places=get_unique(places_array)
	unique_topics=get_unique(topics_array)
	print(unique_places)
	print(unique_topics)
	print('number of empty places objects: ',places_empty)
	print('number of empty topics objects: ',topics_empty)
	
main()