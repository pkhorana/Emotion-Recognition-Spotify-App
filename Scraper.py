import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_Disney_Channel_series"

r = requests.get(URL)

list_urls = []

html_soup = BeautifulSoup(r.content, 'html5lib')


