import argparse
import os
import shutil
import random

from change_xml_file import change_file

def load_files(folder, sample_num_train):
  files_jpg = list()
  files_xml = list()
  files_name = os.listdir('./'+str(folder))
  
  for file_name in files_name:
    if file_name[-3:] == 'jpg':
      files_jpg.append(file_name)
    else:
      files_xml.append(file_name)
  
  load_files_random(folder, sample_num_train, len(files_jpg), [files_jpg, files_xml])


def load_files_random(folder, sample_num_train, indexs, files):
  num_samples_train = round((indexs * sample_num_train)/100)

  new_list = random.sample(range(0, indexs), num_samples_train)
  sambles_test, sambles_train = consult_itens(new_list, indexs, files)
  changing_directories(folder, sambles_train, sambles_test)


def consult_itens(new_list, indexs, files):
  sample_test = list()
  sample_train = list()
  for index in range(indexs):
    if (index not in new_list):
      sample_test.append(files[0][index])
      sample_test.append(files[1][index])

  for i in new_list:
    sample_train.append(files[0][i])
    sample_train.append(files[1][i]) 
  
  return sample_test, sample_train


def changing_directories(folder, samples_train, samples_test):
  path_train = './train'
  path_test = './test'
  if (os.path.isdir(path_train)==False):
        try:
            os.mkdir(path_train)
        except OSError:
            print ("Creation of the directory %s failed" % path_train)
  if (os.path.isdir(path_train)==False):
        try:
            os.mkdir(path_test)
        except OSError:
            print ("Creation of the directory %s failed" % path_test)

  source_dir = folder

  train_dir = os.getcwd() 
  train_dir = train_dir.replace(train_dir, path_train)

  test_dir = os.getcwd() 
  test_dir = test_dir.replace(test_dir, path_test)
     
  for file_name_train in samples_train:
    directory_train = os.path.join(source_dir, file_name_train)
    if directory_train[-3:] == 'xml':
      change_file(directory_train, train_dir, "train")
    shutil.move(directory_train, train_dir)
  
  for file_name_test in samples_test:
    directory_test = os.path.join(source_dir, file_name_test)
    if directory_test[-3:] == 'xml':
      change_file(directory_test, test_dir, "test")
    shutil.move(directory_test, test_dir)

if __name__ == '__main__':
  
  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--folder", type=str, required=True, help="Folder of saved files")
  ap.add_argument("-train", "--train_num", type=int, required=True, help="Total number of training samples")
  args = vars(ap.parse_args())

  folder = args['folder']
  train_num = args['train_num']

  load_files(folder, train_num)
