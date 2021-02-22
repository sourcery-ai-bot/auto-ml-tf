from xml.etree import ElementTree as et

def change_file(directory, path, typ):
  file = et.parse(directory)
  file.find('folder').text = typ
  file.find('path').text = path
  file.write(directory)
    