#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 15:22
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : oversampling.py
# @Software: PyCharm
# @Description 执行过采样

"""
train_SVM.py

VARPA, University of Coruna
Mondejar Guerra, Victor M.
15 Dec 2017
"""

import os
import csv
import gc
# import cPickle as pickle
import pickle
import time
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import collections
from sklearn import svm
import numpy as np

cpu_threads = 7


# http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/combine/plot_comparison_combine.html#sphx-glr-auto-examples-combine-plot-comparison-combine-py

# Perform the oversampling method over the descriptor data
def perform_oversampling(oversample_method, db_path, oversample_features_name, tr_features, tr_labels):
    """
    执行过采样
    :param oversample_method: 过采样函数名
    :param db_path: 数据文件路径
    :param oversample_features_name: 过采样特征名
    :param tr_features: 训练特征
    :param tr_labels: 训练标签
    :return: 过采样后的训练特征和训练标签
    """
    start = time.time()

    oversample_features_pickle_name = db_path + oversample_features_name + '_' + oversample_method + '.p'
    print(oversample_features_pickle_name)

    print("Oversampling method:\t" + oversample_method + " ...")
    # 1 SMOTE
    oversample = None
    if oversample_method == 'SMOTE':
        # kind={'borderline1', 'borderline2', 'svm'}
        svm_model = svm.SVC(C=0.001, kernel='rbf', degree=3, gamma='auto', decision_function_shape='ovo')
        oversample = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5, kind='svm',
                           svm_estimator=svm_model, n_jobs=1)
        # PROBAR SMOTE CON OTRO KIND
    elif oversample_method == 'SMOTE_regular_min':
        oversample = SMOTE(ratio='minority', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                           kind='regular', svm_estimator=None, n_jobs=1)

    elif oversample_method == 'SMOTE_regular':
        oversample = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                           kind='regular', svm_estimator=None, n_jobs=1)

    elif oversample_method == 'SMOTE_border':
        oversample = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                           kind='borderline1', svm_estimator=None, n_jobs=1)

    # 2 SMOTEENN
    elif oversample_method == 'SMOTEENN':
        oversample = SMOTEENN()

    # 3 SMOTE TOMEK
    # NOTE: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.3904&rep=rep1&type=pdf
    elif oversample_method == 'SMOTETomek':
        oversample = SMOTETomek()

    # 4 ADASYN
    elif oversample_method == 'ADASYN':
        oversample = ADASYN(ratio='auto', random_state=None, n_neighbors=5, n_jobs=cpu_threads)

    if oversample is not None:
        tr_features_balanced, tr_labels_balanced = oversample.fit_sample(tr_features, tr_labels)
        # TODO Write data oversampled!
        print("Writing oversampled data at: " + oversample_features_pickle_name + " ...")
        np.savetxt('mit_db/' + oversample_features_name + '_DS1_labels.csv', tr_labels_balanced.astype(int), '%.0f')
        with open(oversample_features_pickle_name, 'wb') as f:
            pickle.dump(tr_features_balanced, f, 2)

        count = collections.Counter(tr_labels_balanced)
        print("Oversampling balance")
        print(count)

        end = time.time()
        print("Time required: " + str(format(end - start, '.2f')) + " sec")

        return tr_features_balanced, tr_labels_balanced
