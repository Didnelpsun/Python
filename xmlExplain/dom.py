from xml.dom.minidom import parse
import xml.dom.minidom

DOMTree = parse("xml/test.xml")
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
    print(collection.getAttribute("shelf"))

movies = collection.getElementsByTagName("movies")
for movie in movies:
    print("***movie***")
    if movie.hasAttribute("title"):
        print("Title: %s" % movie.getAttribute("title"))
    print("Type: %s" % movie.getElementsByTagName("type")[0].childNodes[0].data)
    print("Format: %s" % movie.getElementsByTagName("format")[0].childNodes[0].data)
    print("Rating: %s" % movie.getElementsByTagName("rating")[0].childNodes[0].data)
    print("Stars: %s" % movie.getElementsByTagName("stars")[0].childNodes[0].data)
    print("Description: %s" % movie.getElementsByTagName("description")[0].childNodes[0].data)
