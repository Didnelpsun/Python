import requests

from lxml import etree

url = 'http://qq.com/'
headers = {"User-Agent" : "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)"}
response = requests.get(url = url, headers = headers).text
html = etree.HTML(response, etree.HTMLParser())

lis = html.xpath('//ul//li//a//text()')
for item in lis:
    print(item)