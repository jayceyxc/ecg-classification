#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 15:24
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : train_SVM.py
# @Software: PyCharm
# @Description

"""
train_SVM.py

VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

from python.load_MITBIH import *
from python.evaluation_AAMI import *
from python.aggregation_voting_strategies import *
from python.oversampling import *
from python.cross_validation import *
from python.feature_selection import *

import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn import decomposition

import os


def create_svm_model_name(model_svm_path, winL, winR, do_preprocess,
                          maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection,
                          oversamp_method, leads_flag, reduced_DS, pca_k, delimiter):
    if reduced_DS:
        model_svm_path = model_svm_path + delimiter + 'exp_2'

    if leads_flag[0] == 1:
        model_svm_path = model_svm_path + delimiter + 'MLII'

    if leads_flag[1] == 1:
        model_svm_path = model_svm_path + delimiter + 'V1'

    if oversamp_method:
        model_svm_path = model_svm_path + delimiter + oversamp_method

    if feature_selection:
        model_svm_path = model_svm_path + delimiter + feature_selection

    if do_preprocess:
        model_svm_path = model_svm_path + delimiter + 'rm_bsln'

    if maxRR:
        model_svm_path = model_svm_path + delimiter + 'maxRR'

    if use_RR:
        model_svm_path = model_svm_path + delimiter + 'RR'

    if norm_RR:
        model_svm_path = model_svm_path + delimiter + 'norm_RR'

    for descp in compute_morph:
        model_svm_path = model_svm_path + delimiter + descp

    if use_weight_class:
        model_svm_path = model_svm_path + delimiter + 'weighted'

    if pca_k > 0:
        model_svm_path = model_svm_path + delimiter + 'pca_' + str(pca_k)

    return model_svm_path


def eval_model(svm_model, features, labels, multi_mode, voting_strategy, output_path, C_value, gamma_value, DS):
    """
    Eval the SVM model and export the results
    :param svm_model:
    :param features:
    :param labels:
    :param multi_mode:
    :param voting_strategy:
    :param output_path:
    :param C_value:
    :param gamma_value:
    :param DS:
    :return:
    """
    if multi_mode == 'ovo':
        decision_ovo = svm_model.decision_function(features)

        if voting_strategy == 'ovo_voting':
            predict_ovo, counter = ovo_voting(decision_ovo, 4)

        elif voting_strategy == 'ovo_voting_both':
            predict_ovo, counter = ovo_voting_both(decision_ovo, 4)

        elif voting_strategy == 'ovo_voting_exp':
            predict_ovo, counter = ovo_voting_exp(decision_ovo, 4)

        # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
        perf_measures = compute_AAMI_performance_measures(predict_ovo, labels)

    elif multi_mode == 'ovr':
        decision_ovr = svm_model.decision_function(features)
        predict_ovr = svm_model.predict(features)
        perf_measures = compute_AAMI_performance_measures(predict_ovr, labels)

    # Write results and also predictions on DS2
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if gamma_value != 0.0:
        write_AAMI_results(perf_measures, output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) +
                           '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + voting_strategy + '.txt')
    else:
        write_AAMI_results(perf_measures, output_path + '/' + DS + 'C_' + str(C_value) +
                           '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + voting_strategy + '.txt')

    # Array to .csv
    if multi_mode == 'ovo':
        if gamma_value != 0.0:
            np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) +
                       '_decision_ovo.csv', decision_ovo)
            np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) +
                       '_predict_' + voting_strategy + '.csv', predict_ovo.astype(int), '%.0f')
        else:
            np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) +
                       '_decision_ovo.csv', decision_ovo)
            np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) +
                       '_predict_' + voting_strategy + '.csv', predict_ovo.astype(int), '%.0f')

    elif multi_mode == 'ovr':
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) +
                   '_decision_ovr.csv', decision_ovr)
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) +
                   '_predict_' + voting_strategy + '.csv', predict_ovr.astype(int), '%.0f')

    print("Results writed at " + output_path + '/' + DS + 'C_' + str(C_value))


def create_oversample_name(reduced_DS, do_preprocess, compute_morph, winL, winR, maxRR, use_RR, norm_RR, pca_k):
    """
    拼装上采样名称
    :param reduced_DS:
    :param do_preprocess:
    :param compute_morph:
    :param winL:
    :param winR:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param pca_k:
    :return:
    """
    oversamp_features_pickle_name = ''
    if reduced_DS:
        oversamp_features_pickle_name += '_reduced_'

    if do_preprocess:
        oversamp_features_pickle_name += '_rm_bsline'

    if maxRR:
        oversamp_features_pickle_name += '_maxRR'

    if use_RR:
        oversamp_features_pickle_name += '_RR'

    if norm_RR:
        oversamp_features_pickle_name += '_norm_RR'

    for descp in compute_morph:
        oversamp_features_pickle_name += '_' + descp

    if pca_k > 0:
        oversamp_features_pickle_name += '_pca_' + str(pca_k)

    oversamp_features_pickle_name += '_wL_' + str(winL) + '_wR_' + str(winR)

    return oversamp_features_pickle_name


def main(multi_mode='ovo', winL=90, winR=90, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversample_method='', pca_k='', feature_selection='',
         do_cross_val='', C_value=0.001, gamma_value=0.0, reduced_DS=False, leads_flag=[1, 0]):
    """

    :param multi_mode: the paramter decision_function_shape of the svm.SVC.
                       'ovo', 'ovr', default='ovr'
                       Whether to return a one-vs-rest ('ovr') decision function of shape
                       (n_samples, n_classes) as all other classifiers, or the original
                       one-vs-one ('ovo') decision function of libsvm which has shape
                       (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
                       ('ovo') is always used as multi-class strategy.
    :param winL: 信号采样点的左边窗口大小，表示信号所在采样点的左侧多少个采样点
    :param winR: 信号采样点的右边窗口大小，表示信号所在采样点的右侧多少个采样点
    :param do_preprocess: 是否进行预处理
    :param use_weight_class:
    :param maxRR: 是否使用最大RR间隔
    :param use_RR: 是否使用RR间隔
    :param norm_RR: 是否使用平均的RR间隔
    :param compute_morph: 计算心电特征的变形，心跳信息。
           resample_10: 使用傅立叶变换对心跳信息进行重新取样
           raw: 使用原始的心跳信息
           u-lbp: compute_uniform_LBP
           lbp: compute_LBP
           hbf5: compute_HBF
           wvlt: compute_wavelet_descriptor
           wvlt+pca: compute_wavelet_descriptor + IncrementalPCA
           HOS: compute_hos_descriptor
           myMorph: compute_my_own_descriptor
    :param oversample_method: 过采样方法，可选参数如下：SMOTE、SMOTE_regular_min、SMOTE_regular、SMOTE_border、SMOTEENN、
                              SMOTETomek、ADASYN
    :param pca_k: 是否使用 IncrementalPCA 降维
    :param feature_selection: 特征选择方式。
           select_K_Best：SelectKBest，
           LassoCV：LassoCV，
           select_percentile：SelectPercentile
    :param do_cross_val: 是否进行交叉验证
    :param C_value: The parameter C of the svm.SVC. Penalty parameter C of the error term.
    :param gamma_value: the parameter gama of the svm.SVN.
                        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

                        Current default is 'auto' which uses 1 / n_features,
                        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
                        as value of gamma. The current default of gamma, 'auto', will change
                        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
                        'auto' is used as a default indicating that no explicit value of gamma
                        was passed.
    :param reduced_DS:
    :param leads_flag:
    :return:
    """
    print("Runing train_SVM.py!")

    db_path = '/Users/yuxuecheng/Work/XLTech/documents/产品相关/心电识别/数据集/mitbih_database/'

    # Load train data
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                 reduced_DS, leads_flag)

    # Load Test data
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess,
                                                                       maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                       reduced_DS, leads_flag)
    if reduced_DS:
        np.savetxt('mit_db/' + 'exp_2_' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')
    else:
        np.savetxt('mit_db/' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')

        # if reduced_DS == True:
    #    np.savetxt('mit_db/' + 'exp_2_' + 'DS1_labels.csv', tr_labels.astype(int), '%.0f')
    # else:
    # np.savetxt('mit_db/' + 'DS1_labels.csv', tr_labels.astype(int), '%.0f')

    ##############################################################
    # 0) TODO if feature_Selection:
    # before oversamp!!?????

    # TODO perform normalization before the oversampling?
    if oversample_method:
        # Filename
        oversample_features_pickle_name = create_oversample_name(reduced_DS, do_preprocess, compute_morph, winL, winR,
                                                               maxRR, use_RR, norm_RR, pca_k)

        # Do oversampling
        tr_features, tr_labels = perform_oversampling(oversample_method, db_path + 'oversample/python_mit',
                                                      oversample_features_pickle_name, tr_features, tr_labels)

    # Normalization of the input data
    # scaled: zero mean unit variance ( z-score )
    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)

    # scaled: zero mean unit variance ( z-score )
    eval_features_scaled = scaler.transform(eval_features)
    ##############################################################
    # 0) feature_Selection: also after Oversampling
    if feature_selection:
        print("Runing feature selection")
        best_features = 7
        tr_features_scaled, features_index_sorted = run_feature_selection(tr_features_scaled, tr_labels,
                                                                          feature_selection, best_features)
        eval_features_scaled = eval_features_scaled[:, features_index_sorted[0:best_features]]
    # 1)
    if pca_k > 0:
        # Load if exists??
        # NOTE PCA do memory error!

        # NOTE 11 Enero: TEST WITH IPCA!!!!!!
        start = time.time()

        print("Runing IPCA " + str(pca_k) + "...")

        # Run PCA
        IPCA = sklearn.decomposition.IncrementalPCA(pca_k, batch_size=pca_k)  # gamma_pca

        # tr_features_scaled = KPCA.fit_transform(tr_features_scaled)
        IPCA.fit(tr_features_scaled)

        # Apply PCA on test data!
        tr_features_scaled = IPCA.transform(tr_features_scaled)
        eval_features_scaled = IPCA.transform(eval_features_scaled)

        """
        print("Runing TruncatedSVD (singular value decomposition (SVD)!!!) (alternative to PCA) " + str(pca_k) + "...")

        svd = decomposition.TruncatedSVD(n_components=pca_k, algorithm='arpack')
        svd.fit(tr_features_scaled)
        tr_features_scaled = svd.transform(tr_features_scaled)
        eval_features_scaled = svd.transform(eval_features_scaled)

        """
        end = time.time()

        print("Time runing IPCA (rbf): " + str(format(end - start, '.2f')) + " sec")
    ##############################################################
    # 2) Cross-validation:

    if do_cross_val:
        print("Runing cross val...")
        start = time.time()

        # TODO Save data over the k-folds and ranked by the best average values in separated files
        perf_measures_path = create_svm_model_name(
            '/Users/yuxuecheng/Learn/OpenSource/medicine/ecg-classification/python/results/' + multi_mode, winL, winR,
            do_preprocess,
            maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection, oversample_method, leads_flag,
            reduced_DS, pca_k, '/')

        # TODO implement this method! check to avoid NaN scores....

        if do_cross_val == 'pat_cv':  # Cross validation with one fold per patient
            cv_scores, c_values = run_cross_val(tr_features_scaled, tr_labels, tr_patient_num_beats, do_cross_val,
                                                len(tr_patient_num_beats))

            if not os.path.exists(perf_measures_path):
                os.makedirs(perf_measures_path)
            np.savetxt(perf_measures_path + '/cross_val_k-pat_cv_F_score.csv', (c_values, cv_scores.astype(float)),
                       "%f")

        elif do_cross_val == 'beat_cv':  # cross validation by class id samples
            k_folds = {5}
            for k in k_folds:
                ijk_scores, c_values = run_cross_val(tr_features_scaled, tr_labels, tr_patient_num_beats, do_cross_val,
                                                     k)
                # TODO Save data over the k-folds and ranked by the best average values in separated files
                perf_measures_path = create_svm_model_name(
                    '/Users/yuxuecheng/Learn/OpenSource/medicine/ecg-classification/python/results/' + multi_mode, winL, winR,
                    do_preprocess,
                    maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection, oversample_method,
                    leads_flag, reduced_DS, pca_k, '/')

                if not os.path.exists(perf_measures_path):
                    os.makedirs(perf_measures_path)
                np.savetxt(perf_measures_path + '/cross_val_k-' + str(k) + '_Ijk_score.csv',
                           (c_values, ijk_scores.astype(float)), "%f")

            end = time.time()
            print("Time running Cross Validation: " + str(format(end - start, '.2f')) + " sec")
    else:

        ################################################################################################
        # 3) Train SVM model

        # TODO load best params from cross validation!

        use_probability = False

        model_svm_path = db_path + 'svm_models/' + multi_mode + '_rbf'

        model_svm_path = create_svm_model_name(model_svm_path, winL, winR, do_preprocess,
                                               maxRR, use_RR, norm_RR, compute_morph, use_weight_class,
                                               feature_selection,
                                               oversample_method, leads_flag, reduced_DS, pca_k, '_')

        if gamma_value != 0.0:
            model_svm_path = model_svm_path + '_C_' + str(C_value) + '_g_' + str(gamma_value) + '.joblib.pkl'
        else:
            model_svm_path = model_svm_path + '_C_' + str(C_value) + '.joblib.pkl'

        print("Training model on MIT-BIH DS1: " + model_svm_path + "...")

        if os.path.isfile(model_svm_path):
            # Load the trained model!
            svm_model = joblib.load(model_svm_path)

        else:
            class_weights = {}
            for c in range(4):
                class_weights.update({c: len(tr_labels) / float(np.count_nonzero(tr_labels == c))})

            # class_weight='balanced',
            if gamma_value != 0.0:  # NOTE 0.0 means 1/n_features default value
                svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma=gamma_value,
                                    coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                                    cache_size=200, class_weight=class_weights, verbose=False,
                                    max_iter=-1, decision_function_shape=multi_mode, random_state=None)
            else:
                svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma='auto',
                                    coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                                    cache_size=200, class_weight=class_weights, verbose=False,
                                    max_iter=-1, decision_function_shape=multi_mode, random_state=None)

            # Let's Train!

            start = time.time()
            svm_model.fit(tr_features_scaled, tr_labels)
            end = time.time()
            # TODO assert that the class_ID appears with the desired order,
            # with the goal of ovo make the combinations properly
            print("Trained completed!\n\t" + model_svm_path + "\n \
                \tTime required: " + str(format(end - start, '.2f')) + " sec")

            # Export model: save/write trained SVM model
            joblib.dump(svm_model, model_svm_path)

            # TODO Export StandardScaler()

        #########################################################################
        # 4) Test SVM model
        print("Testing model on MIT-BIH DS2: " + model_svm_path + "...")

        ############################################################################################################
        # EVALUATION
        ############################################################################################################

        # Evaluate the model on the training data
        perf_measures_path = create_svm_model_name(
            '/Users/yuxuecheng/Learn/OpenSource/medicine/ecg-classification/python/results/' + multi_mode, winL, winR,
            do_preprocess,
            maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection, oversample_method, leads_flag,
            reduced_DS, pca_k, '/')

        # ovo_voting:
        # Simply add 1 to the win class
        print("Evaluation on DS1 ...")
        eval_model(svm_model, tr_features_scaled, tr_labels, multi_mode, 'ovo_voting', perf_measures_path, C_value,
                   gamma_value, 'Train_')

        # Let's test new data!
        print("Evaluation on DS2 ...")
        eval_model(svm_model, eval_features_scaled, eval_labels, multi_mode, 'ovo_voting', perf_measures_path, C_value,
                   gamma_value, '')

        # ovo_voting_exp:
        # Consider the post prob adding to both classes
        print("Evaluation on DS1 ...")
        eval_model(svm_model, tr_features_scaled, tr_labels, multi_mode, 'ovo_voting_exp', perf_measures_path, C_value,
                   gamma_value, 'Train_')

        # Let's test new data!
        print("Evaluation on DS2 ...")
        eval_model(svm_model, eval_features_scaled, eval_labels, multi_mode, 'ovo_voting_exp', perf_measures_path,
                   C_value, gamma_value, '')
