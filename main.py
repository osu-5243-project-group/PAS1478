from bs4 import BeautifulSoup
from glob import glob
import math
import numpy as np

def get_filenames():
	filenames=glob('*.sgm')
	return filenames

def load_data(filenames):
	data=[]
	fields=['date','people','orgs','exchanges','companies','title','dateline','body','unknown']
	labels=['topics','places']
	for filename in filenames:
		soup = BeautifulSoup(open(filename), 'html.parser')
		reuters=soup.find_all('reuters')
		for article in reuters:
			line=[]
			for field in fields:
				temp=article.find(field)
				if(temp!=None):
					line.append(temp.string)
				else:
					line.append('empty')
			for label in labels:
				temp_labels=[]
				temp=article.find(label)
				ds=temp.find_all('d')
				for d in ds:
					temp_labels.append(d.string)
				line.append(temp_labels)
			if(len(line[9])> 0 and len(line[10])>0):
				data.append(line)
	print(len(data))
	return data

#split into train/test/validation data
def split_data(train_percent,test_percent,validate_percent,data):
	total=10596
	train_data=[]
	test_data=[]
	validate_data=[]
	print('splitting data into ',train_percent,' train data, ',test_percent,' test data,',validate_percent,' validate data')
	train_index=math.floor(train_percent*total)
	test_index=math.floor(test_percent*total)
	validate_index=math.floor(validate_percent*total)
	train_data=data[0:train_index+1]
	test_data=data[train_index+2:train_index+2+test_index]
	validate_data=data[train_index+3+test_index:train_index+3+test_index+validate_index]
	print(len(train_data))
	print(len(test_data))
	print(len(validate_data))
	return train_data,test_data,validate_data

#train
def train():
	articles=['the','a','an']
#test
def test():
	print('test')
#validate
def validate():
	print('validate')

def main():
	text_tags=[]
	filenames=get_filenames()
	data=load_data(filenames)
	train_data,test_data,validate_data=split_data(0.60,0.30,0.10,data)

main()