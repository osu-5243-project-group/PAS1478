from bs4 import BeautifulSoup
from glob import glob

def get_filenames():
	filenames=glob('*.sgm')
	return filenames
	
def get_places(soup):
	places=soup.find_all('places')
	for place in places:
		ds=place.find_all('d')
		for d in ds:
			print(d.string)

def get_topics(soup):
	topics=soup.find_all('topics')
	for topic in topics:
		ds=topic.find_all('d')
		for d in ds:
			print(d.string)
		
def main():
	filenames=get_filenames()
	soup = BeautifulSoup(open(filenames[0]), 'html.parser')
	get_places(soup)
	get_topics(soup)
	
main()