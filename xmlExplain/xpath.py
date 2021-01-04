from lxml import etree

xml = etree.parse('xml/data.xml')
movies = xml.xpath('//movies/@title')

for element in movies:
    print(element)

# years = xml.xpath('//movies/year/text()')
# for element in years:
#     print(element)
#
# result = xml.xpath('//movies/format/text()')
# for element in result:
#     print(element)
