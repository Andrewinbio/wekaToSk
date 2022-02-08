#	imports:

from os.path import isdir
from os.path import listdir
from os.path import abspath, dirname, exists
from sys import argv

import argparse
#import sklearn.utils.resample
import sklearn.utils
import configparser #this is for reading the properties file on lines 69-77
import pandas as pd
import arff #documentation for this: https://pythonhosted.org/liac-arff/
#from scipy.io.arff import loadarff
# The following replaces [import weka.classifers.*] and [import weka.classifiers.meta.*]

import sklearn.naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
###import needed for rules.PART
from sklearn.neighbors import KNeighborsClassifier

##need an alternative for weka.core.*
##need an alternative for weka.core.converters.ConverterUtils.DataSource

#feature_selection replaces weka.filters
import sklearn.feature_selection.*


def dump(instances, filename):
	w = open(filename, 'w')
	w.write(str(instances))
	w.write('\n')
#	w.flush() this doesn't have to be here since python automatically does this before closing a filename
	w.close()



def balance(instances)
	#	use sklearn.utils.resample here
	#	use sklearn.feature_selection here in place of Filter from weka


#parse options

parentDir = dirname(abspath(argv[0]))
rootDir = dirname(abspath(argv[1]))
currentFold = dirname(abspath(argv[2]))
currentBag = int(argv[3])
attr_imp_bool = bool(argv[4])

inputFilename = rootDir + "/data.arff"

classifierString = argv[5:-1]
classifierName = classifierString[0]
shortClassifierName = classifierName.split("\\.")[-1]
classifierOptions = []
if (len(classifierString) > 1):
	classifierOptions = classifierString[1:-1]


# load data parameters from properties file
p = configparser.ConfigParser()
p.read('sk.properties') #formerly weka.properties
workingDir = rootDir + "/" + p.get("sk", ".")
idAttribute = p.get("sk", "idAttribute")
classAttribute = p.get("sk", "classAttribute")
balanceTraining = bool(p.get("sk", "balanceTraining"))
balanceTest = bool(p.get("sk", "balanceTest")) #this parameter doesn't appear to be in existing properties files but I have it here anyways
classLabel = p.get("sk", "classLabel")

assert p.has_option("sk", "foldCount") or p.has_option("sk", "foldAttribute")

if (p.has_option("foldCount")):
	foldCount = int(p.get("sk", "foldCount"))

foldAttribute = p.get("sk", "foldAttribute")
nestedFoldCount = int(p.get("sk", "nestedFoldCount"))
bagCount = int(p.get("bagCount"))
writeModel = bool(p.get("sk", "writeModel"))

# load data, determine if regression or classification
source = arff.load(open(inputFilename)) # the arff is now a dictionary
data = pd.DataFrame(source['data']) #data stored in pandas dataframe
regression = isinstance(source['data'][0][0], float) or isinstance(source['data'][0][0], int) #checks the data to see if it is numeric

if (not regression):
	predictClassValue = p.get("sk", "predictClassValue")

#shuffle data, set class variable
data = shuffle(data) #shuffles data without replacement
setattr(data, 'type', classAttribute) # I am unsure if this is a valid alternative to data.setClass(data.attribute(classAttribute)) 
#pd.DataFrame([q.val for q in data], columns = [classAttribute] )
if (not regression):
	predictClassIndex = data[data[classAttribute] == predictClassValue].index
	assert predictClassIndex != -1
	print ("[%s] %s, generating probabilities for class %s (index %d)\n" %(shortClassifierName, classAttribute, predictClassValue, predictClassIndex))
	
else:
	print("[%s] %s, generating predictions\n" %(shortClassifierName, classAttribute))

#add ids if not specified

if (idAttribute == ""):
    idAttribute = "ID"

