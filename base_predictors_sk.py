from os import makedirs
from os.path import abspath, exists
from time import time
import pickle
import configparser
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
import common
from sklearn.utils import shuffle, resample
import os
from sklearn.inspection import permutation_importance
import argparse
from importlib.util import spec_from_file_location, module_from_spec


random_seed = 42

def split_train_test_by_fold(fold_col_exist, data_df, fold_col, current_fold, clf_name, fold_count, y_col):
    idx = pd.IndexSlice
    if fold_col_exist:
        print("[%s] generating %s folds for leave-one-value-out CV\n" % (clf_name, fold_count))
        fold_count = len(data_df[fold_col].unique())
        fold_outertestbool = (data_df[fold_col] == current_fold)
        test = data_df.loc[fold_outertestbool]
        train = data_df.loc[~fold_outertestbool]
    else:  # train test split is done here
        print("[%s] generating folds for %s-fold CV \n" % (clf_name, fold_count))
        y = data_df[y_col]
        kFold = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
        kf_nth_split = list(kFold.split(data_df, y))[current_fold]
        fold_mask = np.array(range(data_df.shape[0])) == kf_nth_split[1]
        test = data_df.loc[fold_mask]
        train = data_df.loc[~fold_mask]
    return train, test, fold_count


def multiidx_dataframe_balance_sampler(dataf, y_col):
    # UnderSampling majority label
    rus = RandomUnderSampler(random_state=random_seed)
    y = dataf[y_col]
    resampled_df, _ = rus.fit_resample(dataf, y)
    return pd.DataFrame(data=resampled_df, columns=dataf.columns)


def multiidx_dataframe_resampler_wr(dataf):
    # Resample with replacement
    resampled_df = resample(dataf, random_state=random_seed)
    return pd.DataFrame(data=resampled_df, columns=dataf.columns)


def balance_or_resample(dataf_train, dataf_test, bag_count,
                        regression_bool, bl_training_bool,
                        bl_test_bool, clf_name, current_bag, y_col):
    if bag_count > 0:
        # TODO: with replacement
        print(" [%s] generating bag %d\n" % (clf_name, current_bag))
        dataf_train = multiidx_dataframe_resampler_wr(dataf_train)
    if not regression_bool and bl_training_bool:
        print("[%s] balancing training samples \n" % classifierName)
        dataf_train = multiidx_dataframe_balance_sampler(dataf_train, y_col)
    if not regression_bool and bl_test_bool:
        print("[%s] balancing test samples\n" % classifierName)
        dataf_test = multiidx_dataframe_balance_sampler(dataf_test, y_col)
    return dataf_train, dataf_test


def split_df_X_y_idx(dataf, nonfeat_cols, id_col, y_col, reg_bool, pred_class_val):
    X = dataf.drop(columns=nonfeat_cols)
    y = dataf.loc[:, y_col]
    indices = dataf.loc[:, id_col]
    return X, y, indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base predictor skip')
    parser.add_argument('--parentDir', type=str, required=True, help='Path of parent')
    parser.add_argument('--rootDir', type=str, required=True, help='Path of root')
    parser.add_argument('--currentFold', type=str, required=True, help='Outer fold')
    parser.add_argument('--currentBag', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
    parser.add_argument('--attr_imp_bool', type=common.str2bool, default=False, help='Run feature importance or not')
    parser.add_argument('--classifierName', type=str, required=True, help='Name of the classifier in classifier.py')
    args = parser.parse_args()

    # parse options
    parentDir = abspath(args.parentDir)
    rootDir = abspath(args.rootDir)
    currentFold = int(args.currentFold)
    currentBag = args.currentBag
    attr_imp_bool = args.attr_imp_bool

    inputFilename = os.path.join(rootDir, "data.arff")
    data = common.read_arff_to_pandas_df(inputFilename)

    classifierName = args.classifierName

    # load data parameters from properties file
    p = configparser.ConfigParser()
    p.read(os.path.join(parentDir, 'sk.properties'))
    p_sk = p['sk']
    workingDir = rootDir
    idAttribute = p_sk.get("idAttribute")
    classAttribute = p_sk.get("classAttribute")
    balanceTraining = common.str2bool(p_sk.get("balanceTraining"))
    balanceTest = common.str2bool(p_sk.get("balanceTest"))
    classLabel = p_sk.get("classLabel")

    assert ("foldCount" in p_sk) or ("foldAttribute" in p_sk)

    foldAttribute = p_sk.get("foldAttribute")

    if "foldAttribute" in p_sk:
        foldCount = len(data[foldAttribute].unique())
    elif "foldCount" in p_sk:
        foldCount = int(p_sk.get("foldCount"))

    nestedFoldCount = int(p_sk.get("nestedFoldCount"))
    bagCount = int(p_sk.get("bagCount"))
    if not p_sk.get("writeModel") is None:
        writeModel = common.str2bool(p_sk.get("writeModel"))
    else:
        writeModel = None

    regression = len(data[classAttribute].unique()) > 2  # checks the data to see if it is numeric

    if not regression:
        predictClassValue = p_sk.get("predictClassValue")

    # shuffle data, set class variable
    data = shuffle(data, random_state=random_seed)  # shuffles data without replacement

    foldAttribute_exist = ('foldAttribute' in p_sk)
    index_cols = [idAttribute, classAttribute]
    if foldAttribute_exist:
        data[foldAttribute] = data[foldAttribute].astype(str)
        index_cols.append(foldAttribute)
        currentFold = str(currentFold)

    if not regression:
        y_ = data[classAttribute]
        y_.loc[~(y_ == predictClassValue)] = 0
        y_.loc[y_ == predictClassValue] = 1
        data[classAttribute] = y_.astype(int)

    outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
                                                                  data_df=data,
                                                                  fold_col=foldAttribute,
                                                                  current_fold=currentFold,
                                                                  clf_name=classifierName,
                                                                  fold_count=foldCount,
                                                                  y_col=classAttribute
                                                                  )

    # resample and balance training of fold if necessary
    outer_train, outer_test = balance_or_resample(dataf_train=outer_train,
                                                  dataf_test=outer_test,
                                                  bag_count=bagCount,
                                                  regression_bool=regression,
                                                  bl_training_bool=balanceTraining,
                                                  bl_test_bool=balanceTest,
                                                  clf_name=classifierName,
                                                  current_bag=currentBag,
                                                  y_col=classAttribute)

    print("[%s] fold: %s bag: %s training size: %d test size: %d\n" % (
        classifierName, currentFold, "none" if (bagCount == 0) else currentBag, outer_train.shape[0],
        outer_test.shape[0]))
    start = time()

    # import predictors
    predictors_path = parentDir + '/define_base_predictors.py'
    assert exists(predictors_path)
    bp_spec = spec_from_file_location('predictors', predictors_path)
    bp_module = module_from_spec(bp_spec)
    bp_spec.loader.exec_module(bp_module)
    predictors = bp_module.predictors
    classifier = predictors[classifierName]

    outer_train_X, outer_train_y, outer_train_id = split_df_X_y_idx(outer_train,
                                                                    nonfeat_cols=index_cols,
                                                                    y_col=classAttribute,
                                                                    id_col=idAttribute,
                                                                    reg_bool=regression,
                                                                    pred_class_val=predictClassValue
                                                                    )
    classifier.fit(X=outer_train_X, y=outer_train_y)

    duration = time() - start
    durationMinutes = duration / (1e3 * 60)
    print("[%s] trained in %.2f minutes, evaluating\n" % (classifierName, durationMinutes))

    # write predictions to csv
    classifierDir = os.path.join(workingDir, 'base-predictor-' + classifierName)
    if not exists(classifierDir):
        makedirs(classifierDir, exist_ok=True)

    outputPrefix = "predictions-%s-%02d" % (currentFold, currentBag)
    if writeModel:
        pickle.dump(classifier, open(os.path.join(classifierDir, outputPrefix + ".sav", 'wb')))

    output_cols = ["id", "label", "prediction", "fold", "bag", "classifier"]
    outer_test_X, outer_test_y, outer_test_id = split_df_X_y_idx(outer_test,
                                                                 nonfeat_cols=index_cols,
                                                                 y_col=classAttribute,
                                                                 id_col=idAttribute,
                                                                 reg_bool=regression,
                                                                 pred_class_val=predictClassValue
                                                                 )
    outer_test_prediction = common.generic_classifier_predict(clf=classifier,
                                                              regression_bool=regression,
                                                              input_data=outer_test_X
                                                              )

    outer_test_result_df = pd.DataFrame({'id': outer_test[idAttribute],
                                         'label': outer_test_y,
                                         'prediction': outer_test_prediction,
                                         'fold': outer_test[foldAttribute]})

    outer_test_result_df['bag'] = currentBag
    outer_test_result_df['classifier'] = classifierName

    outer_test_result_df.to_csv(os.path.join(classifierDir, outputPrefix + '.csv.gz'), compression='gzip', index=False)

    if nestedFoldCount == 0:
        SystemExit

    outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
                                                                  data_df=data,
                                                                  fold_col=foldAttribute,
                                                                  current_fold=currentFold,
                                                                  clf_name=classifierName,
                                                                  fold_count=foldCount,
                                                                  y_col=classAttribute
                                                                  )

    outer_train_X, outer_train_y, outer_train_id = split_df_X_y_idx(outer_train,
                                                                    nonfeat_cols=index_cols,
                                                                    y_col=classAttribute,
                                                                    id_col=idAttribute,
                                                                    reg_bool=regression,
                                                                    pred_class_val=predictClassValue
                                                                    )

    inner_cv_kf = StratifiedKFold(n_splits=nestedFoldCount, shuffle=True, random_state=random_seed)
    inner_cv_split = inner_cv_kf.split(outer_train, outer_train_y)
    for currentNestedFold, (inner_train_idx, inner_test_idx) in enumerate(inner_cv_split):
        inner_test = outer_train.iloc[inner_test_idx, :]
        inner_train = outer_train.iloc[inner_train_idx, :]

        inner_train, inner_test = balance_or_resample(dataf_train=inner_train,
                                                      dataf_test=inner_test,
                                                      bag_count=bagCount,
                                                      regression_bool=regression,
                                                      bl_training_bool=balanceTraining,
                                                      bl_test_bool=balanceTest,
                                                      clf_name=classifierName,
                                                      current_bag=currentBag,
                                                      y_col=classAttribute)

        print("[{} inner {}] fold: {} bag: {} training size: {} test size: {}\n".format(classifierName,
                                                                                        currentNestedFold,
                                                                                        currentFold,
                                                                                        currentBag,
                                                                                        inner_train.shape[0],
                                                                                        inner_test.shape[0]))

        start = time()
        classifier = predictors[classifierName]
        inner_train_X, inner_train_y, inner_train_id = split_df_X_y_idx(inner_train,
                                                                        nonfeat_cols=index_cols,
                                                                        y_col=classAttribute,
                                                                        id_col=idAttribute,
                                                                        reg_bool=regression,
                                                                        pred_class_val=predictClassValue
                                                                        )
        classifier.fit(X=inner_train_X, y=inner_train_y)
        inner_test_X, inner_test_y, inner_test_id = split_df_X_y_idx(inner_test,
                                                                     nonfeat_cols=index_cols,
                                                                     y_col=classAttribute,
                                                                     id_col=idAttribute,
                                                                     reg_bool=regression,
                                                                     pred_class_val=predictClassValue
                                                                     )
        inner_test_prediction = common.generic_classifier_predict(clf=classifier,
                                                                  regression_bool=regression,
                                                                  input_data=inner_test_X
                                                                  )
        end = time()
        time_spent = end - start
        print("[{} inner {}] trained and evaluated in {:2f} minutes".format(classifierName, currentNestedFold,
                                                                            time_spent / 60))

        outputPrefix = "validation-%s-%02d-%02d.csv.gz" % (currentFold, currentNestedFold, currentBag)
        nested_cols = ['id', 'label', 'prediction', 'fold', 'nested_fold', 'bag', 'classifier']
        result_df = pd.DataFrame({'id': inner_test[idAttribute],
                                  'label': inner_test_y,
                                  'prediction': inner_test_prediction,
                                  'fold': inner_test[foldAttribute],
                                  })
        result_df['nested_fold'] = currentNestedFold
        result_df['bag'] = currentBag
        result_df['classifier'] = classifierName

        result_df.to_csv(os.path.join(classifierDir, outputPrefix), compression='gzip', index=False)

    # Jamie's code
    # Attribute Importance
    if attr_imp_bool:
        outer_train, outer_test, foldCount = split_train_test_by_fold(fold_col_exist=foldAttribute_exist,
                                                                      data_df=data,
                                                                      fold_col=foldAttribute,
                                                                      current_fold=currentFold,
                                                                      clf_name=classifierName,
                                                                      fold_count=foldCount,
                                                                      y_col=classAttribute
                                                                      )
        attribute_importance = dict(permutation_importance(estimator=classifier,
                                                           X=outer_train_X,
                                                           y=outer_train_y,
                                                           n_jobs=-1))
        outputPrefix = "attribute_imp-%s-%02d" % (currentFold, currentBag)
        importances = attribute_importance.pop("importances")
        attribute_importances_df = pd.DataFrame.from_dict(attribute_importance)
        for i in range(importances.shape[-1]):  # loop over importance permutation results
            attribute_importances_df[f"importance_{i + 1}"] = importances[:, i]  # add importances as new columns
        attribute_importances_df.currentFold = currentFold  # attach additional info as metadata
        attribute_importances_df.currentBag = currentBag
        attribute_importances_df.shortClassifierName = classifierName

        attribute_importances_df.to_csv(os.path.join(classifierDir, outputPrefix), index=False)
