# a very simple file to downlaod and reproduce some of the results 
# from the original context tree paper: 

import requests 
from os import exists
from libsvm.svmutil import svm_read_problem

if not exists('data/classification_dataset.csv'):
    classification_dataset_url  = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing" # noqa E501
    classification_dataset = requests.get(classification_dataset_url).content
    # write file locally if it does not already exist 
    with open('data/classification_dataset.csv', 'wb') as file:
        file.write(classification_dataset)

y, X = svm_read_problem('data/classification_dataset.csv')
