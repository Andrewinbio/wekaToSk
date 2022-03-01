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
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Random Forest
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.linear_model import LogisticRegression, LinearRegression  # LR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor  # Adaboost
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Decision Tree
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # Logit Boost with parameter(loss='deviance')
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # KNN

from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier, XGBRegressor # XGB
from sklearn.svm import SVC, LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.model_selection import KFold
import common
from sklearn.utils import shuffle, resample
import os
from sklearn.inspection import permutation_importance
import argparse

##need an alternative for weka.core.*
##need an alternative for weka.core.converters.ConverterUtils.DataSource

#feature_selection replaces weka.filters
import sklearn.feature_selection

random_seed = 42

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

def split_train_test_by_fold(fold_col_exist,data_df,fold_col,current_fold, clf_name, fold_count):
	idx = pd.IndexSlice
	if fold_col_exist:
		fold_count = len(data_df[fold_col].unique())
		fold_outertestbool = (data_df[fold_col]==current_fold)
		print(fold_outertestbool)
		test = data_df.loc[fold_outertestbool]
		# test = data_df.iloc[fold_outertestbool, :]
		train = data_df.loc[fold_outertestbool]
		print("[%s] generating %s folds for leave-one-value-out CV\n" % (clf_name, fold_count))
	else:  # train test split is done here
		print("[%s] generating folds for %s-fold CV \n" % (clf_name, fold_count))
		kFold = KFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
		kf_nth_split = list(kFold.split(data_df))[current_fold]
		fold_mask = np.array(range(data_df.shape[0])) == kf_nth_split[1]
		# test = data_df.iloc[kf_nth_split[1], :]
		# train = data_df.iloc[kf_nth_split[0], :]
		test = data_df.loc[fold_mask]
		train = data_df.loc[~fold_mask]

	return train, test, fold_count

def multiidx_dataframe_balance_sampler(dataf, y_col):
	# UnderSampling majority label
	rus = RandomUnderSampler(random_state=random_seed)
	# X_resampled, y_resampled = rus.fit_resample(train.values, train[classAttribute])
	# Create a numeric index to for undersampler, which will be used to index the dataframe
	# numeric_df_index = dataf.index.get_level_values(idAttribute).values
	# y = dataf.index.get_level_values(classAttribute).values
	numeric_df_index = dataf.index.values
	y = dataf.loc[:,y_col]
	numeric_df_index_resampled, _ = rus.fit_resample(numeric_df_index, y)
	# print(numeric_df_index_resampled.shape)
	# numeric_df_index_resampled
	return dataf.loc[numeric_df_index_resampled].reset_index()

def multiidx_dataframe_resampler_wr(dataf):
	# Resample with replacement
	# numeric_df_index = dataf.index.get_level_values(idAttribute)
	resampled_df = resample(dataf, random_state=random_seed)

	# print(numeric_df_index_resampled)
	# return dataf.loc[numeric_df_index_resampled]
	return pd.DataFrame(data=resampled_df, columns=dataf.columns)

def balance_or_resample(dataf_train, dataf_test, bag_count,
						regression_bool, bl_training_bool,
						bl_test_bool, clf_name, current_bag):
	if (bag_count > 0):
		# TODO: with replacement
		print(" [%s] generating bag %d\n" % (clf_name, current_bag))
		dataf_train = multiidx_dataframe_resampler_wr(dataf_train)
		print(dataf_train)

	if ((not regression_bool) and bl_training_bool):
		print("[%s] balancing training samples \n" % (classifierName))
		dataf_train = multiidx_dataframe_balance_sampler(dataf_train)

	if ((not regression_bool) and bl_test_bool):
		print("[%s] balancing test samples\n" % (classifierName))
		dataf_test = multiidx_dataframe_balance_sampler(dataf_test)

	return dataf_train, dataf_test

def split_df_X_y_idx(dataf, nonfeat_cols, id_col, y_col):
	X = dataf.drop(columns=nonfeat_cols)
	y = dataf.loc[:,y_col]
	indices = dataf.loc[:,id_col]
	return X, y, indices


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Base predictor skip')
	parser.add_argument('--parentDir', type=str, required=True, help='Path of parent')
	parser.add_argument('--rootDir', type=str, required=True, help='Path of root')
	parser.add_argument('--currentFold', type=str, required=True, help='Outer fold')
	parser.add_argument('--currentBag', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
	parser.add_argument('--attr_imp_bool', type=common.str2bool, default=False, help='Run Feature importance or not')
	parser.add_argument('--classifierName', type=str, required=True, help='Name of the classifier in classifier.py')
	args = parser.parse_args()

	#parse options
	parentDir = abspath(args.parentDir)
	print(parentDir)
	rootDir = abspath(args.rootDir)
	currentFold = args.currentFold
	currentBag = args.currentBag
	attr_imp_bool = args.attr_imp_bool

	inputFilename = os.path.join(rootDir, "data.arff")

	# classifierString = argv[5:-1]
	classifierName = args.classifierName

	# shortClassifierName = classifierName.split("\\.")[-1]
	# classifierOptions = []
	# if (len(classifierString) > 1):
	# 	classifierOptions = classifierString[1:-1]


	# load data parameters from properties file
	p = configparser.ConfigParser()
	p.read(os.path.join(parentDir,'sk.properties')) #formerly weka.properties
	print(p['sk'])
	p_sk = p['sk']
	workingDir = rootDir
	idAttribute = p_sk.get("idAttribute")
	classAttribute = p_sk.get("classAttribute")
	balanceTraining = bool(p_sk.get("balanceTraining"))
	balanceTest = bool(p_sk.get("balanceTest")) #this parameter doesn't appear to be in existing properties files but I have it here anyways
	classLabel = p_sk.get("classLabel")

	assert ("foldCount" in p_sk) or ("foldAttribute" in p_sk)

	if ("foldCount" in p_sk):
		foldCount = int(p_sk.get("foldCount"))
	else:
		foldCount = None

	foldAttribute = p_sk.get("foldAttribute")
	nestedFoldCount = int(p_sk.get("nestedFoldCount"))
	bagCount = int(p_sk.get("bagCount"))
	writeModel = bool(p_sk.get("writeModel"))

	# load data, determine if regression or classification
	# source = arff.load(open(inputFilename)) # the arff is now a dictionary
	# replace by the custom function to load arff
	data = common.read_arff_to_pandas_df(inputFilename) #
	# data = pd.DataFrame(source['data']) #data stored in pandas dataframe
	# regression = isinstance(data['data'][0][0], float) or isinstance(source['data'][0][0], int) #checks the data to see if it is numeric
	regression = len(data[classAttribute].unique()) > 2  #checks the data to see if it is numeric

	if (not regression):
		predictClassValue = p_sk.get("predictClassValue")

	#shuffle data, set class variable
	data = shuffle(data, random_state=random_seed) #shuffles data without replacement

	foldAttribute_exist = (foldAttribute != "")
	index_cols = [idAttribute, classAttribute]
	if foldAttribute_exist:
		data[foldAttribute] = data[foldAttribute].astype(str)
		index_cols.append(foldAttribute)

	# data.set_index(index_cols, inplace=True)

	# setattr(data, 'type', classAttribute) # I am unsure if this is a valid alternative to data.setClass(data.attribute(classAttribute))
	#pd.DataFrame([q.val for q in data], columns = [classAttribute] )
	print(data)
	# if (not regression):
	# 	# predictClassIndex = data[data.loc[classAttribute] == predictClassValue].index
	# 	predictClassIndex = data[data.loc[classAttribute] == predictClassValue].index
	# 	assert predictClassIndex != -1
	# 	print ("[%s] %s, generating probabilities for class %s (index %d)\n" %(classifierName, classAttribute, predictClassValue, predictClassIndex))
	#
	# else:
	# 	print("[%s] %s, generating predictions\n" %(classifierName, classAttribute))

	#add ids if not specified
	# if (idAttribute == ""):
	# 	#id's are automatically speciified however as the data is in a
	# 	#pandas data frame as opposed to prior state in java
	# 	idAttribtue = data.index

	#generate folds
	# if (foldAttribute != ""):
	# if foldAttribute_exist:
	# 	foldCount = len(data[foldAttribute].unique())
	# 	fold_outertestbool = (data[foldAttribute] == currentFold)
	# 	outer_test = data.loc[fold_outertestbool, :]
	# 	outer_train = data.loc[~fold_outertestbool, :]
	# 	# foldAttibuteIndex = str(data[foldAttribute].index + 1)
	# 	# foldAttributeValueIndex = str(data[data[foldAttribute] == currentFold].index + 1)
	# 	print("[%s] generating %s folds for leave-one-value-out CV\n" %(classifierName,foldCount))
	# 	# *****need to add equivalents of lines 123 to 137 from base.groovy here*****
	# 	#X_train, X_test, Y_train, Y_test = train_test_split(data[:-1],data[-1], test_size = 0.2)
	# 	#sklearn.cross_validation.KFold(n= int(data.shape[0]), n_folds=foldCount, shuffle=False, random_state=None)
	#
	# 	# prelootest = pd.concat([Y_train, Y_test])
	# 	# prelootrain = pd.concat([X_train, X_test], axis=1)
	#
	# 	# lootest = LeaveOneOut().get_n_splits(prelootest)
	# 	# lootrain = LeaveOneOut.get_n_splits(prelootrain)
	#
	# 	# test = data[lootest]
	# 	# train = data[lootrain]
	#
	# 	# X = data.values
	# 	# Y = data[classAttribute]
	#
	# 	# loo = LeaveOneOut()
	# 	# logo = LeaveOneGroupOut()
	#
	# 	# for train_index, test_index in logo.split(data[:-1]):
	# 	# 	looX_train, looX_test = X[train_index], X[test_index]
	# 	# 	looY_train, looY_test = Y[train_index], Y[test_index]
	# 	#
	# 	# test = pd.concat([looY_train, looY_test])
	# 	# train = pd.concat([looX_train, looX_test], axis=1)
	#
	#
	#
	# else: #train test split is done here
	# 	print("[%s] generating folds for %s-fold CV \n" %(classifierName, foldCount))
	#
	# 	# X_train, X_test, Y_train, Y_test = train_test_split(data[:-1],data[-1], test_size = 0.2)
	# 	# #sklearn.cross_validation.KFold(n= int(data.shape[0]), n_folds=foldCount, shuffle=False, random_state=None)
	# 	# prektest = pd.concat([Y_train, Y_test])
	# 	# prektrain = pd.concat([X_train, X_test], axis=1)
	#
	# 	# ktest = KFold(n_splits = foldCount).get_n_splits(prektest)
	# 	# ktrain = KFold(n_splits = foldCount).get_n_splits(prektrain)
	#
	# 	# test = data[ktest]
	# 	# train = data[ktrain]
	#
	# 	# X = data[:-1]
	# 	# Y = data [-1]
	#
	# 	kFold = KFold(n_splits=foldCount, random_state=random_seed)
	# 	kf_nth_split = list(kFold.split(data))[currentFold]
	# 	outer_test = data.loc[kf_nth_split[1],:]
	# 	outer_train = data.loc[kf_nth_split[0],:]

		# for train_index, test_index in kFold.split(data[:-1]):
		# 	kFoldX_train, kFoldX_test = X[train_index], X[test_index]
		# 	kFoldY_train, kFoldY_test = Y[train_index], Y[test_index]
		#
		# test = pd.concat([kFoldY_train, kFoldY_test])
		# train = pd.concat([kFoldX_train, kFoldX_test], axis=1)

	outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
													   data_df=data,
													   fold_col=foldAttribute,
													   current_fold=currentFold,
													   clf_name=classifierName,
													   fold_count=foldCount
													   )

	#resample and balance training of fold if necessary
	outer_train, outer_test = balance_or_resample(dataf_train=outer_train,
												  dataf_test=outer_test,
												  bag_count=bagCount,
												  regression_bool=regression,
												  bl_training_bool=balanceTraining,
												  bl_test_bool=balanceTest,
												  clf_name=classifierName,
												  current_bag=currentBag)
	# if (bagCount > 0):
	# 	print(" [%s] generating bag %d\n" %(classifierName,currentBag))
	# 	#train = train.resample(random.randrange(currentBag)) #unsure if the newRandom(currentbag)) argument is necessary
	# 	rus = RandomUnderSampler(random_state=random_seed)
	# 	# X_resampled, y_resampled = rus.fit_resample(train.values, train[classAttribute])
	# 	# Create a numeric index to for undersampler, which will be used to index the dataframe
	# 	numeric_train_index = range(outer_train.shape[0])
	# 	numeric_train_index_resampled, outer_y_train_resampled = rus.fit_resample(numeric_index_X, outer_train[classAttribute])
	# 	outer_train = outer_train.iloc[numeric_train_index_resampled,:]
	#
	# if((not regression) and balanceTraining):
	# 	print("[%s] blancing training samples \n" %(classifierName))
	# 	outer_train = balance(outer_train)
	#
	# if((not regression) and balanceTest):
	# 	print("[%s] balancing test samples\n" %(classifierName))
	# 	outer_test = balance(outer_test)

	# init filtered classifier
	#classifier (as Abstract Classifier was a class that all
	# weka classifiers are built upon this is no longer needed for
	# sklearin) and removeFilter no longer needed

	# lines 159-172 equivalent no longer needed from base_predictors.groovy

	# train, store duration
	print("[%s] fold: %s bag: %s training size: %d test size: %d\n" %(classifierName, currentFold, "none"  if (bagCount == 0) else currentBag, outer_train.shape[0], outer_test.shape[0]))
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
	outer_train_X, outer_train_y, outer_train_id = split_df_X_y_idx(outer_train,
																	nonfeat_cols=index_cols,
																	y_col=classAttribute,
																	id_col=idAttribute)
	classifier.fit(X=outer_train_X, y=outer_train_y)

	duration = time() - start
	durationMinutes = duration / (1e3 * 60)
	print ("[%s] trained in %.2f minutes, evaluating\n" %(classifierName, durationMinutes))

	# write predictions to csv
	classifierDir = os.path.join(workingDir, 'base-predictor-'+classifierName)
	if (not exists(classifierDir)):
		mkdir(classifierDir, exist_ok=True)

	outputPrefix = "predictions-%s-%02d" %(currentFold, currentBag)

	# writer = open(classifierDir + outputPrefix + ".csv", 'w')
	#writer = csv.writer(open(classifierDir + outputPrefix + ".csv", 'w'))
	#writer = csv.writer(open(classifierDir + outputPrefix + ".csv.gz", 'w'))
	# *****need to gzip this*****
	if (writeModel):
		pickle.dump(classifier, open(os.path.join(classifierDir,outputPrefix + ".sav", 'wb')))


	# header = print("# %s@%s %.2f minutes\n" %(os.path.expanduser, socket.gethostname(), durationMinutes))
	#writer = csv.writer(outFile)
	# writer.write("header")
	output_cols = ["id","label","prediction","fold","bag","classifier"]
	outer_test_prediction = common.generic_classifier_predict(clf=classifier,
															  regression_bool=regression,
															  input_data=outer_test.values
															  )

	outer_test_result_df = pd.DataFrame({'id':outer_test[idAttribute],
										 'label':outer_test[classAttribute],
										 'prediction': outer_test_prediction,
										 'fold':outer_test[foldAttribute]})

	outer_test_result_df['bag'] = currentBag
	outer_test_result_df['classifier'] = classifierName
	# outer_test_result_df.to_csv(os.path.join(classifierDir, outputPrefix + 'csv'), index=False)
	outer_test_result_df.to_csv(os.path.join(classifierDir, outputPrefix + '.csv.gz'),compression='gzip', index=False)
	# writer.write("id,label,prediction,fold,bag,classifier\n")
	# for instance in test:
	# 	id = str(data[idAttribtue])
	# 	if (not regression): #I don't think the if and else statements are no longer needed here
	# 		label = instance[-1]
	# 		prediction = classifier.predict(instance[:-1])
	# 	else:
	# 		label = float(instance[-1])
	# 		prediction = classifier.predict(instance[:-1])
	# row = print("%s,%s,%f,%s,%s,%s\n" %(id, label, prediction, currentFold, currentBag, shortClassifierName))
	# writer.write(row)
	#
	# writer.flush()
	# writer.close()

	if (nestedFoldCount == 0):
		SystemExit

	# Richard:

	outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
													   data_df=data,
													   fold_col=foldAttribute,
													   current_fold=currentFold,
													   clf_name=classifierName,
													   fold_count=foldCount
													   )

	inner_cv_kf = KFold(n_splits=nestedFoldCount,shuffle=True, random_state=random_seed)
	inner_cv_split = inner_cv_kf.split(outer_train)
	for currentNestedFold, (inner_train_idx, inner_test_idx) in enumerate(inner_cv_split):
		inner_test = outer_train.iloc[inner_test_idx,:]
		inner_train = outer_train.iloc[inner_train_idx,:]

		inner_train, inner_test = balance_or_resample(dataf_train=inner_train,
													  dataf_test=inner_test,
													  bag_count=bagCount,
													  regression_bool=regression,
													  bl_training_bool=balanceTraining,
													  bl_test_bool=balanceTest,
													  clf_name=classifierName,
													  current_bag=currentBag)

		print("[{} inner {}] fold: {} bag: {} training size: {} test size: {}\n".format(classifierName,
																						currentNestedFold,
																						currentFold,
																						currentBag,
																						inner_train.shape[0],
																						inner_test.shape[0]))

		start = time()
		classifier = classifiers.get(classifierName)
		inner_train_X, inner_train_y, inner_train_id = split_df_X_y_idx(inner_train,
																		nonfeat_cols=index_cols,
																		y_col=classAttribute,
																		id_col=idAttribute)
		classifier.fit(X=inner_train_X, y=inner_train_y)
		# classifier.fit(inner_train.values, inner_train.index.get_level_values(classAttribute))
		inner_test_prediction = common.generic_classifier_predict(clf=classifier,
																  regression_bool=regression,
																  input_data=inner_test.values
																  )
		end = time()
		time_spent = end - start
		print("[{} inner {}] trained and evaluated in {:2f} minutes".format(classifierName, currentNestedFold, time_spent/60))

		outputPrefix = "validation-%s-%02d-%02d.csv.gz" % (currentFold, currentNestedFold, currentBag)
		# outputPrefix = "validation-%s-%02d-%02d.csv" % (currentFold, currentNestedFold, currentBag)
		nested_cols = ['id','label','prediction','fold','nested_fold','bag','classifier']
		result_df = pd.DataFrame({'id': inner_test[idAttribute],
								  'label': inner_test[classAttribute],
								  'prediction':inner_test_prediction,
								  'fold': inner_test[foldAttribute],
								  })
		result_df['nested_fold'] = currentNestedFold
		result_df['bag'] = currentBag
		result_df['classifier'] = classifierName

		result_df.to_csv(os.path.join(classifierDir, outputPrefix),compression='gzip', index=False)
		# result_df.to_csv(os.path.join(classifierDir, outputPrefix), index=False)

	# Jamie's code
	# Attribute Importance
	if attr_imp_bool:
		outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
																	  data_df=data,
																	  fold_col=foldAttribute,
																	  current_fold=currentFold,
																	  clf_name=classifierName,
																	  fold_count=foldCount
																	  )
		attribute_importance = dict(permutation_importance(estimator=classifierName,
														   X=outer_train.values,
														   y=outer_train[classAttribute],
														   njobs=-1))
		# print(f{classifierName})
		# Export Attribute Importance as pandas DataFrame
		outputPrefix = "attribute_imp-%s-%02d" % (currentFold, currentBag)
		# classifierDir, outputPrefix + ".csv.gz")
		importances = attribute_importance.pop("importances")  # importances are vector valued so remove and add as rows
		attribute_importances_df = pd.DataFrame.from_dict(attribute_importance)
		for i in range(importances.shape[-1]):  # loop over importance permutation results
			attribute_importances_df[f"importance_{i + 1}"] = importances[:, i]  # add importances as new columns
		attribute_importances_df.currentFold = currentFold  # attach additional info as metadata
		attribute_importances_df.currentBag = currentBag
		attribute_importances_df.shortClassifierName = classifierName

		attribute_importances_df.to_csv(os.path.join(classifierDir, outputPrefix), index=False)