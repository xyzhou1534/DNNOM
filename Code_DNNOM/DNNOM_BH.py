'''
Author: Zhou Xiangyu
Date: 2026-02-01 16:12
E-mail: xyzhou1534@gmail.com
'''
# import diy packages
from __api_experiments import (get_classifier,write_excel,now_time_str,mean_std,
                               load_binary_data,load_kto2_classes_data_lm,load_k_classes_data,get_metrics)
import _smote_variants_v1_original
import _smote_variants_v2_Wsmote
import _smote_variants_v3_SW
import _smote_variants_v4_WRND 
import _smote_variants_v5_Gometric
import _smote_variants_v6_GBsmote 
import _smote_variants_v7_OIRP_BO
# from _api_v9_OIRP_BU import PSO_Denoise
from __api_DNNOM_BU import PSO_Denoise, opt_n


# import python packages
import inspect
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.pipeline import make_pipeline
from imblearn import over_sampling
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')
import math
from RSDS import RSDS_zhou
from NaN import NaN_RD
import timeout_decorator


# @timeout_decorator.timeout(1800)    # timeout seconds
def resamplings(data_train, methods):
    data_train_resampleds:dict = {}
    # ntree = int(math.log(data_train.shape[0]*data_train.shape[1]-1,2)/2)
    # RData,weight_SW = RSDS_zhou(data_train, ntree) #去噪后的数据集，CRF权重矩阵
    # NData, weight_WNND, nans = NaN_RD(data_train)  # 原始数据，相对密度，自然邻居
    X,y = data_train[:, 1:], data_train[:, 0]
    n_opt, sampling_strategy = opt_n(X,y)
    print('最佳采样比例:\t',sampling_strategy)
    
    for method,model in methods.items():
        over, under = model[0], model[1]    # class
        if over == SMOTENC:continue

        # original hybrid
        over_model, under_model = over(), under()   # instance of class
        hybrid_model = make_pipeline(over_model, under_model)

        # # OIRP_BH hybrid
        over_model_BH = over(sampling_strategy=sampling_strategy)
        under_model_BH = under()
        hybrid_model_BH = make_pipeline(over_model_BH, under_model_BH)
        try:
            X_res, y_res= hybrid_model.fit_resample(X,y)
            X_res_BH, y_res_BH = hybrid_model_BH.fit_resample(X,y)
        except Exception as e:
            continue
        y_res = y_res.reshape(y_res.shape[0], 1)
        data_train_resampled = np.hstack((y_res, X_res))
        data_train_resampleds[method] = data_train_resampled

        y_res_BH = y_res_BH.reshape(y_res_BH.shape[0], 1)
        data_train_resampled = np.hstack((y_res_BH, X_res_BH))
        data_train_resampleds[method+'_OIRP_BH'] = data_train_resampled
        print('Hybrid_sampled:\t',method)
        print('Hybrid_sampled:\t',method+'_OIRP_BH')

    return data_train_resampleds


def run(result_file_path, data_names, classifiers, methods:dict, 
        noise_rate, is_binary:bool, is_raise:bool,
        # over_models,under_models,
        ):
    table_head = ['data', 'Classifier', 'sampling method', 'noise rate', 'samples', 'features', 'train 0',
                'train 1', 'test 0', 'test 1', 'sampled 0', 'sampled 1', 'accuracy', 'precision', 'recall', 'AUC',
                'f1', 'g_mean', 'right_rate_min', 'tp', 'fp', 'fn', 'tn', 'now_time']
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                            'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']
    imbs = {'splice': 5, 'breastcancer': 5, 'creditApproval': 5, 'votes': 5, 'Online_Retail': 50,
            'codrna': 20, 'nomao': 20, 'mocap': 40, 'poker': 50, 'skin': 100, 'covtype': 60, 'comedy': 100}

    debug = []
    for d, data_name in enumerate(data_names):
        # 丢掉部分数据，使数据达到给定不平衡率（uci_extended 中的数据不作处理）
        imb = imbs[data_name] if data_name in imbs.keys() else (
            0 if data_name in uci_extended else 10)
        if is_binary == 1:   # binary datasets
            data_train, data_test, counters = load_binary_data(
                data_name, noise_rate=noise_rate)  
        elif is_binary == 2:  # multiclass to binary
                data_train, data_test, counters = load_kto2_classes_data_lm(
                data_name, noise_rate=noise_rate)  
        elif is_binary == 0:   # multiclass
            data_train, data_test,counters = load_k_classes_data(
                data_name, noise_rate=noise_rate)

        data_train = np.vstack([data_train, data_test])

        # if counters[0] < 1000:continue  # TODO:调整数据集样本数量
        print('\n\nnoise rate: {:.2f}  data set {:2d}/{:2d}: {:15s}  samples: {:6d}   features: {:3d}   '
            'train 0: {:6d}   train 1: {:6d}   test 0: {:6d}   test 1: {:6d}\n'
            .format(noise_rate, d + 1, len(data_names), data_name,
                    counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]))

        try:
            data_train_resampleds:dict = resamplings(data_train, methods=methods)
        except Exception as e:
            if is_raise:raise e
            else:
                debug.append([noise_rate, data_name, str(e)])
                continue

        for classifier in classifiers:
            print('    classifier: {:10s}'.format(classifier))
            clf = get_classifier(classifier)

            for method in methods.keys():
                try:
                    data_train_those = data_train_resampleds[method]
                except Exception as e:
                    if is_raise:raise e
                    else:
                        debug.append([noise_rate, data_name, method,
                                    classifier, str(e)])
                        continue
                # 5 fold cross validation
                metri = Counter({})
                skf = StratifiedShuffleSplit(n_splits=5)

                try:
                    for train_index, validate_index in skf.split(data_train_those[:, 1:], data_train_those[:, 0]):
                        train, validate = data_train_those[train_index], data_train_those[validate_index]
                        metrics = get_metrics(clf, train, validate)
                        metri += Counter(metrics)
                    metrics = {k: round(v/5, 5)
                               for k, v, in metri.items()}   # 平均每折的值

                    # record the results to csv
                    excel_line = [
                        data_name, classifier, method, noise_rate,
                        counters[0], counters[1], counters[2], counters[3], counters[4], counters[5],
                        Counter(train[:, 0])[0], Counter(train[:, 0])[1],
                        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['auc'],
                        metrics['f1'], metrics['g_mean'], now_time_str()
                    ]

                    write_excel(result_file_path, [], table_head=table_head)
                    print('{}  {:20s}'.format(now_time_str(), method),'\t\t', metrics)
                    write_excel(result_file_path, [excel_line], table_head=table_head)
                except Exception as e:
                    if is_raise:raise e
                    else:
                        debug.append([noise_rate, data_name, method, classifier, str(e)])
                        continue
    # save debug json
    df = pd.DataFrame(debug)
    df.to_csv(r'KBsomte/SMOTE/debug/'+now_time_str()+'.csv', index=False)


def main(is_binary:bool,is_raise:bool):
    if is_binary:
        data_names = [
            'abalone-21_vs_8', 'abalone-3_vs_11', 'abalone9-18', 'cleveland-0_vs_4', 
            'dermatology-6', 'ecoli-0-1-3-7_vs_2-6', 'ecoli-0-1-4-6_vs_5', 'glass-0-1-4-6_vs_2', 
            'glass-0-1-6_vs_2', 'led7digit-0-2-4-5-6-7-8-9_vs_1', 'lymphography-normal-fibrosis', 
            'page-blocks-1-3_vs_4', 'poker-9_vs_7', 'shuttle-6_vs_2-3', 'shuttle-c2-vs-c4', 
            'winequality-red-3_vs_5', 'winequality-red-8_vs_6', 'yeast-0-3-5-9_vs_7-8', 'zoo-3'
        ]
    else:
        data_names = [
            'nuclear', 'contraceptive', 'satimage', 'sensorReadings', 'frogs'
        ]

    classifiers = [
        'KNN',
        'DTree',
        'LR',
        'XGBoost', 
        'LightGBM',
        'SVM',
        'AdaBoost',
        'GBDT',
    ]
    under_models:list = BaseUnderSampler.__subclasses__()
    over_models:list = inspect.getmembers(over_sampling,inspect.isclass)
    models_dict= {name+'_'+under.__name__:(over,under) for name, over 
                            in over_models for under in under_models}

    begin_time = now_time_str(colon=False)
    result_file_path = r'res/{}_BH.xls'.format('1_'+begin_time)

    # for n_r in [1, 3, 5, 7, 9]:
    #     run(result_file_path, data_names, classifiers, methods=models_dict, 
    #             noise_rate=n_r*0.05,is_binary=is_binary,is_raise=is_raise,)
    
    run(result_file_path, data_names, classifiers, methods=models_dict, 
                noise_rate=0.05,is_binary=is_binary,is_raise=is_raise,)

if __name__ == '__main__':
    main(is_binary=1,is_raise=0)