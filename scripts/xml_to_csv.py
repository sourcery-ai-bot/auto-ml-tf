import argparse
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
# Comando - python xml_to_csv.py -i data/images/test -o data/annotations/test_labels.csv
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, 
    help='Path to folder containing the XML files to be converted ')
ap.add_argument('-o', '--output', type=str, required=True, 
    help='Path to output folder for files csv')
ap.add_argument('-f', '--file', type=str, required=True, 
    help='name of the data type ie training or testing')
args = vars(ap.parse_args())

inputs = args['input']
outputs = args['output']
file = args['file']
# inputs ='./'
# outputs = './'
# folder = 'imagens'

xml_df = xml_to_csv(inputs)
xml_df.to_csv((outputs+file+'_labels.csv'), index=None)
print('Successfully converted xml to csv [file {}].'.format(file))
print('Successfully file salved in {}_labels.csv.'.format(file))