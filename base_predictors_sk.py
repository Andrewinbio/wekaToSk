from os import listdir, mkdir
from os.path import abspath, dirname, exists, join, isdir, expanduser
from socket import socket
from sys import argv
from time import time
import csv
import gzip
import pickle

import argparse
# import sklearn.utils.resample
import sklearn.utils
import configparser #this is for reading the properties file on lines 69-77
import pandas as pd
import arff #documentation for this: https://pythonhosted.org/liac-arff/
#from scipy.io.arff import loadarff
# The following replaces [import weka.classifers.*] and [import weka.classifiers.meta.*]

import numpy as np
from random import random
from imlearn.under_sampling import RandomUnderSampler

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

##need an alternative for weka.core.*
##need an alternative for weka.core.converters.ConverterUtils.DataSource

#feature_selection replaces weka.filters
import sklearn.feature_selection


def dump(instances, filename):
	w = open(filename, 'w')
	w.write(str(instances))
	w.write('\n')
#	w.flush() this doesn't have to be here since python automatically does this before closing a filename
	w.close()


#the balance function below serves the purpose of subsampling
# of a random uniform distribution of 
# the data and is later used on the train and test sets
def balance(instances):
	tempnp = instances.to_numpy()
	tempnp.random.uniform(low = 0.0, high = 1.0) #uni
	return pd.DataFrame(tempnp)
	

	#return instances.ix[random.sample(instances.index, frac = 0.60)] # I am unsure if this is too big of a sample
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
	#id's are automatically speciified however as the data is in a 
	#pandas data frame as opposed to prior state in java
	idAttribtue = data.index
	
#generate folds
if (foldAttribute != ""):
	foldCount = data[foldAttribute].value_counts()
	foldAttibuteIndex = str(data[foldAttribute].index + 1)
	foldAttributeValueIndex = str(data[data[foldAttribute] == currentFold].index + 1)
	print("[%s] generating %s folds for leave-one-value-out CV\n" %(shortClassifierName,foldCount))
	# *****need to add equivalents of lines 123 to 137 from base.groovy here*****
	#X_train, X_test, Y_train, Y_test = train_test_split(data[:-1],data[-1], test_size = 0.2)
	#sklearn.cross_validation.KFold(n= int(data.shape[0]), n_folds=foldCount, shuffle=False, random_state=None)
	
	# prelootest = pd.concat([Y_train, Y_test])
	# prelootrain = pd.concat([X_train, X_test], axis=1)
	
	# lootest = LeaveOneOut().get_n_splits(prelootest)
	# lootrain = LeaveOneOut.get_n_splits(prelootrain)

	# test = data[lootest]
	# train = data[lootrain]

	X = data[:-1]
	Y = data [-1]

	loo = LeaveOneOut()

	for train_index, test_index in loo.split(data[:-1]):
		looX_train, looX_test = X[train_index], X[test_index]
		looY_train, looY_test = Y[train_index], Y[test_index]

	test = pd.concat([looY_train, looY_test])
	train = pd.concat([looX_train, looX_test], axis=1)
	
	

else: #train test split is done here
	print("[%s] generating folds for %s-fold CV \n" %(shortClassifierName, foldCount))
	
	# X_train, X_test, Y_train, Y_test = train_test_split(data[:-1],data[-1], test_size = 0.2)
	# #sklearn.cross_validation.KFold(n= int(data.shape[0]), n_folds=foldCount, shuffle=False, random_state=None)
	# prektest = pd.concat([Y_train, Y_test])
	# prektrain = pd.concat([X_train, X_test], axis=1)

	# ktest = KFold(n_splits = foldCount).get_n_splits(prektest)
	# ktrain = KFold(n_splits = foldCount).get_n_splits(prektrain)
	
	# test = data[ktest]
	# train = data[ktrain]

	X = data[:-1]
	Y = data [-1]

	kFold = KFold(n_splits=foldCount)

	for train_index, test_index in kFold.split(data[:-1]):
		kFoldX_train, kFoldX_test = X[train_index], X[test_index]
		kFoldY_train, kFoldY_test = Y[train_index], Y[test_index]

	test = pd.concat([kFoldY_train, kFoldY_test])
	train = pd.concat([kFoldX_train, kFoldX_test], axis=1)
	
	#resample and balance training of fold if necessary
if (bagCount > 0):
	print(" [%s] generating bag %d\n" %(shortClassifierName,currentBag))
	#train = train.resample(random.randrange(currentBag)) #unsure if the newRandom(currentbag)) argument is necessary
	rus = RandomUnderSampler(random_state=0)
	X_resampled, y_resampled = rus.fit_resample(train[:-1], train[-1])
	train = pd.concat([X_resampled,y_resampled], axis=1)
	
if((not regression) and balanceTraining):
	print("[%s] blancing training samples \n" %(shortClassifierName))
	train = balance(train)

if((not regression) and balanceTest):
	print("[%s] balancing test samples\n" %(shortClassifierName))
	test = balance(test)

# init filtered classifier
#classifier (as Abstract Classifier was a class that all 
# weka classifiers are built upon this is no longer needed for 
# sklearin) and removeFilter no longer needed

# lines 159-172 equivalent no longer needed from base_predictors.groovy

# train, store duration
print("[%s] fold: %s bag: %s training size: %d test size: %d\n" %(shortClassifierName, currentFold, "none"  if (bagCount == 0) else currentBag, train.numInstances(), test.numInstances()))
start = time()

#*******need to build classifier here*******

classifiers = {
					"RF": RandomForestClassifier(),
					"SVM": SVC(kernel='linear', probability=True),
					"NB": GaussianNB(),
					"LR": LogisticRegression(),
					"AdaBoost": AdaBoostClassifier(),
					"DT": DecisionTreeClassifier(),
					"GradientBoosting": GradientBoostingClassifier(),
					"KNN": KNeighborsClassifier(),
					"XGB": XGBClassifier()
				}
classifier = classifiers.get(classifierName)
	

duration = time() - start
durationMinutes = duration / (1e3 * 60)
print ("[%s] trained in %.2f minutes, evaluating\n" %(shortClassifierName, durationMinutes))

# write predictions to csv
classifierDir = os.path.join(workingDir, classifierName)
if (not exists(classifierDir)):
	mkdir(classifierDir)

outputPrefix = print("priedictions-%s-%02d" %(currentFold, currentBag))

writer = open(classifierDir + outputPrefix + ".csv", 'w')
#writer = csv.writer(open(classifierDir + outputPrefix + ".csv", 'w'))
#writer = csv.writer(open(classifierDir + outputPrefix + ".csv.gz", 'w'))
# *****need to gzip this*****
if(writeModel):
	pickle.dump(classifier, open(classifierDir + outputPrefix + ".sav", 'wb'))


header = print("# %s@%s %.2f minutes %s\n" %(os.path.expanduser, socket.gethostname(), durationMinutes, classifierString.join(" "))) 
#writer = csv.writer(outFile)
writer.write("header")
writer.write("id,label,prediction,fold,bag,classifier\n")
for instance in test:
	id = str(data[idAttribtue])
	if (not regression): #I don't think the if and else statements are no longer needed here
		label = instance[-1]
		prediction = classifier.predict(instance[:-1])
	else:
		label = float(instance[-1])
		prediction = classifier.predict(instance[:-1])
row = print("%s,%s,%f,%s,%s,%s\n" %(id, label, prediction, currentFold, currentBag, shortClassifierName))
writer.write(row)

writer.flush()
writer.close()

if (nestedFoldCount == 0):
	SystemExit