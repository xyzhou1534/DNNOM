
# 画折线图时的特殊设置
framework_methods_spe = ['SMOTE', 'SMOTE-Ge', 'SMOTE-W', 'SMOTE-SW', 'SMOTE-GB',
                    'SMOTE_ENN', 'SMOTE_ENN-Ge', 'SMOTE_ENN-W', 'SMOTE_ENN-SW','SMOTE_ENN-GB',
                    'SMOTE_TomekLinks', 'SMOTE_TomekLinks-Ge', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-SW', 'SMOTE_TomekLinks-GB',
                    'SMOTE_IPF', 'SMOTE_IPF-Ge', 'SMOTE_IPF-W', 'SMOTE_IPF-SW','SMOTE_IPF-GB',]

# 测试SMOTE
# framework_methods = ['SMOTE', 'SMOTE-G', 'SMOTE-W', 'SMOTE-SW', 'SMOTE-GB',
#                     'SMOTE_ENN', 'SMOTE_ENN-G', 'SMOTE_ENN-W', 'SMOTE_ENN-SW','SMOTE_ENN-GB',
#                     'SMOTE_TomekLinks', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-SW', 'SMOTE_TomekLinks-GB',
#                     'SMOTE_IPF', 'SMOTE_IPF-G', 'SMOTE_IPF-W', 'SMOTE_IPF-SW','SMOTE_IPF-GB',]

# framework_methods = ['SMOTE-GB', 'SMOTE', 'SMOTE-G', 'SMOTE-W', 'SMOTE-SW',
#                      'SMOTE_ENN-GB', 'SMOTE_ENN', 'SMOTE_ENN-G', 'SMOTE_ENN-W', 'SMOTE_ENN-SW',
#                      'SMOTE_TomekLinks-GB', 'SMOTE_TomekLinks', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-SW',
#                      'SMOTE_IPF-GB', 'SMOTE_IPF', 'SMOTE_IPF-G', 'SMOTE_IPF-W', 'SMOTE_IPF-SW', ]



framework_methods = [
                        # 'ADASYN_ClusterCentroids', 'ADASYN_ClusterCentroids_OIRP_BH', 
                        # 'ADASYN_NearMiss', 'ADASYN_NearMiss_OIRP_BH', 
                        # 'ADASYN_RandomUnderSampler', 'ADASYN_RandomUnderSampler_OIRP_BH', 
                        'BorderlineSMOTE_ClusterCentroids', 'BorderlineSMOTE_ClusterCentroids_OIRP_BH', 
                        'BorderlineSMOTE_NearMiss', 'BorderlineSMOTE_NearMiss_OIRP_BH', 
                        'BorderlineSMOTE_RandomUnderSampler', 'BorderlineSMOTE_RandomUnderSampler_OIRP_BH', 
                        # 'KMeansSMOTE_ClusterCentroids', 'KMeansSMOTE_ClusterCentroids_OIRP_BH', 
                        # 'KMeansSMOTE_NearMiss', 'KMeansSMOTE_NearMiss_OIRP_BH', 
                        # 'KMeansSMOTE_RandomUnderSampler', 'KMeansSMOTE_RandomUnderSampler_OIRP_BH', 
                        'RandomOverSampler_ClusterCentroids', 'RandomOverSampler_ClusterCentroids_OIRP_BH', 
                        'RandomOverSampler_NearMiss', 'RandomOverSampler_NearMiss_OIRP_BH', 
                        'RandomOverSampler_RandomUnderSampler', 'RandomOverSampler_RandomUnderSampler_OIRP_BH', 
                        'SMOTE_ClusterCentroids', 'SMOTE_ClusterCentroids_OIRP_BH', 
                        'SMOTE_NearMiss', 'SMOTE_NearMiss_OIRP_BH', 
                        'SMOTE_RandomUnderSampler', 'SMOTE_RandomUnderSampler_OIRP_BH', 
                        'SMOTEN_ClusterCentroids', 'SMOTEN_ClusterCentroids_OIRP_BH', 
                        'SMOTEN_NearMiss', 'SMOTEN_NearMiss_OIRP_BH', 
                        'SMOTEN_RandomUnderSampler', 'SMOTEN_RandomUnderSampler_OIRP_BH', 
                        'SVMSMOTE_ClusterCentroids', 'SVMSMOTE_ClusterCentroids_OIRP_BH', 
                        'SVMSMOTE_NearMiss', 'SVMSMOTE_NearMiss_OIRP_BH', 
                        'SVMSMOTE_RandomUnderSampler', 'SVMSMOTE_RandomUnderSampler_OIRP_BH'
]





framework_multi = ['SMOTE', 'SMOTE-GB', 'SMOTE_ENN', 'SMOTE_ENN-GB',
                    'SMOTE_TomekLinks', 'SMOTE_TomekLinks-GB', 'SMOTE_IPF', 'SMOTE_IPF-GB',]
# SMOTE_IPF 效果比SMOTE-GB好
vsothers_multi = ['SMOTE-GB', 'SMOTE', 'MWMOTE', 'SMOTE_TomekLinks', 'RSMOTE', 'kmeans-smote',  'GBSY', ]

# # 在目前的情况下，ADASYN比SMOTE效果还差，就不作为对比算法了
# vsothers = ['SMOTE-GB', 'SMOTE',  'SMOTE_TomekLinks', 'ADASYN', 'MWMOTE', 'SMOTE_IPF','SMOTE_FRST_2T', 'kmeans-smote', 'RSMOTE', 'AdaptiveSMOTE', 'GDO', ]

# 最终版本
vsothers = ['SMOTE-GB','SMOTE', 'MWMOTE', 'AdaptiveSMOTE', 'SMOTE_TomekLinks', 'SMOTE_IPF', 'RSMOTE', 'kmeans-smote', 'GDO', 'GBSY', ]

framework_wilcoxon = ['SMOTE', 'SMOTE-G', 'SMOTE-W', 'SMOTE-SW', 'SMOTE_ENN', 'SMOTE_ENN-G', 'SMOTE_ENN-W', 'SMOTE_ENN-SW',
                      'SMOTE_TomekLinks', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-SW', 'SMOTE_IPF', 'SMOTE_IPF-G', 'SMOTE_IPF-W', 'SMOTE_IPF-SW', ]

# others_wilcoxon = ['SMOTE', 'SMOTE_TomekLinks', 'ADASYN', 'MWMOTE', 'SMOTE_IPF', 'SMOTE_FRST_2T',
#                    'kmeans-smote', 'RSMOTE', 'AdaptiveSMOTE', 'GDO', ]
# 框架对比中已经由原始的与SMOTE与SMOTE-GB的比较了
others_wilcoxon = ['MWMOTE', 'AdaptiveSMOTE',  'SMOTE_TomekLinks', 'SMOTE_IPF', 
                'RSMOTE', 'kmeans-smote', 'GDO', ]

Classifiers = ['AdaBoost', 'GBDT', 'BPNN', 'DTree', 'LR', 'KNN']
# Classifiers = ['GBDT','DTree', 'AdaBoost', 'KNN', 'LR', 'LightGBM','BPNN', 'SVM']

# Metrics = ['Precision', 'Recall', 'F1-measure', 'AUC', 'G-mean', ]
Metrics = ['precision','recall','f1', 'AUC', 'g_mean', ]

Noise_rates = [0.0, 0.1, 0.2, 0.3]

data_name = {
    # 'zoo':'zoo',
    'new-thyroid1': 'new-thyroid1', 
    'ecoli': 'ecoli', 
    'magic': 'magic', 
    'avila': 'avila', 
    'creditApproval': 'creditApproval', 
    'messidor_features': 'messidor_features', 
    'vehicle2': 'vehicle2', 
    'vehicle': 'vehicle', 
    'page-blocks0': 'page-blocks0', 
    'breastcancer': 'breastcancer', 
    'wilt': 'wilt', 
    'glass0': 'glass0', 
    'pima': 'pima', 
    'wisconsin': 'wisconsin', 
    'flare-F': 'flare-F', 
    'kr-vs-k-zero_vs_eight': 'kr-vs-k', 
    'winequality-red-8_vs_6': 'winequality', 
    
    
    
# 'abalone-21_vs_8': 'abalone-21_vs_8', 
# 'abalone-3_vs_11': 'abalone-3_vs_11', 
# 'abalone9-18': 'abalone9-18', 
# 'cleveland-0_vs_4': 'cleveland', 
# 'dermatology-6': 'dermatology', 
# 'ecoli-0-1-3-7_vs_2-6': 'ecoli-0-1-3-7_vs_2-6', 
# 'ecoli-0-1-4-6_vs_5': 'ecoli-0-1-4-6_vs_5', 
# 'glass-0-1-4-6_vs_2': 'glass-0-1-4-6_vs_2', 
# 'glass-0-1-6_vs_2': 'glass-0-1-6_vs_2', 
# 'led7digit-0-2-4-5-6-7-8-9_vs_1': 'led7digit', 
# 'lymphography-normal-fibrosis': 'lymphography', 
# 'page-blocks-1-3_vs_4': 'page-blocks', 
# 'poker-9_vs_7': 'poker', 
# 'shuttle-6_vs_2-3': 'shuttle-6_vs_2-3', 
# 'shuttle-c2-vs-c4': 'shuttle-c2-vs-c4', 
# 'winequality-red-3_vs_5': 'winequality-red-3_vs_5', 
# 'winequality-red-8_vs_6': 'winequality-red-8_vs_6', 
# 'yeast-0-3-5-9_vs_7-8': 'yeast', 
# 'zoo-3': 'zoo', 
}

method_name = {
                        # 'ADASYN_ClusterCentroids': 'ADASYN_CC', 
                        # 'ADASYN_NearMiss': 'ADASYN_NM',
                        # 'ADASYN_RandomUnderSampler': 'ADASYN_RUS', 
                        'BorderlineSMOTE_ClusterCentroids': 'BS._Cc.', 
                        'BorderlineSMOTE_NearMiss': 'BS._Nm.', 
                        'BorderlineSMOTE_RandomUnderSampler': 'BS._Rus.', 
                        # 'KMeansSMOTE_ClusterCentroids', 'KMeansSMOTE_ClusterCentroids_OIRP_BH', 
                        # 'KMeansSMOTE_NearMiss', 'KMeansSMOTE_NearMiss_OIRP_BH', 
                        # 'KMeansSMOTE_RandomUnderSampler', 'KMeansSMOTE_RandomUnderSampler_OIRP_BH', 
                        'RandomOverSampler_ClusterCentroids': 'Ros._Cc.', 
                        'RandomOverSampler_NearMiss': 'Ros._Nm.', 
                        'RandomOverSampler_RandomUnderSampler': 'Ros._Rus.', 
                        'SMOTE_ClusterCentroids': 'S._Cc.', 
                        'SMOTE_NearMiss': 'S._Nm.', 
                        'SMOTE_RandomUnderSampler': 'S._Rus.', 
                        'SMOTEN_ClusterCentroids': 'SN._Cc.', 
                        'SMOTEN_NearMiss': 'SN._Nm.', 
                        'SMOTEN_RandomUnderSampler': 'SN._Rus.', 
                        'SVMSMOTE_ClusterCentroids': 'SvS._Cc.', 
                        'SVMSMOTE_NearMiss': 'SvS._Nm.', 
                        'SVMSMOTE_RandomUnderSampler': 'SvS._Rus.', 
}



data_name_multi = {
    'wine': 'wine',
    'vertebralColumn': 'verteb',
    'OBS': 'OBS',
    'PhishingData': 'Phishi',
    'abalone': 'abalon',
    'sensorReadings': 'sensor',
    'frogs': 'frogs',
}

Metrics_dict = {
    # 'precision': 'Precision',
    # 'recall': 'Recall',
    'f1': 'F1-measure',
    'auc': 'AUC',
    'g_mean': 'G-mean',
}

framework_color_dict = {
    # 'SMOTE': 'darkcyan',         # 蓝色
    # 'SMOTE-Ge': '#ffa600',      # 橙色
    # 'SMOTE-W': '#58508d',       # 紫色
    # 'SMOTE-SW': '#ff6361',      # 绿色
    # 'SMOTE-GB': '#C82423',      # 红色

    # 'SMOTE_ENN': 'darkcyan',
    # 'SMOTE_ENN-Ge': '#ffa600',
    # 'SMOTE_ENN-W': '#58508d',
    # 'SMOTE_ENN-SW': '#ff6361',
    # 'SMOTE_ENN-GB': '#C82423',

    # 'SMOTE_TomekLinks': 'darkcyan',      # 淡蓝
    # 'SMOTE_TomekLinks-Ge': '#ffa600',
    # 'SMOTE_TomekLinks-W': '#58508d',
    # 'SMOTE_TomekLinks-SW': '#ff6361',
    # 'SMOTE_TomekLinks-GB': '#C82423',

    # 'SMOTE_IPF': 'darkcyan',
    # 'SMOTE_IPF-Ge': '#ffa600',
    # 'SMOTE_IPF-W': '#58508d',
    # 'SMOTE_IPF-SW': '#ff6361',
    # 'SMOTE_IPF-GB': '#C82423',

    'SMOTE': '#59A95A',         # 蓝色
    'SMOTE-Ge': '#2878B5',      # 橙色
    'SMOTE-W': '#58508d',       # 紫色
    # 'SMOTE-SW': '#ff6361',      # 绿色
    'SMOTE-SW': '#F7903D',      # 绿色
    'SMOTE-GB': '#C82423',      # 红色

    'SMOTE_ENN': '#59A95A',
    'SMOTE_ENN-Ge': '#2878B5',
    'SMOTE_ENN-W': '#58508d',
    'SMOTE_ENN-SW': '#F7903D',
    'SMOTE_ENN-GB': '#C82423',

    'SMOTE_TomekLinks': '#59A95A',      # 淡蓝
    'SMOTE_TomekLinks-Ge': '#2878B5',
    'SMOTE_TomekLinks-W': '#58508d',
    'SMOTE_TomekLinks-SW': '#F7903D',
    'SMOTE_TomekLinks-GB': '#C82423',

    'SMOTE_IPF': '#59A95A',
    'SMOTE_IPF-Ge': '#2878B5',
    'SMOTE_IPF-W': '#58508d',
    'SMOTE_IPF-SW': '#F7903D',
    'SMOTE_IPF-GB': '#C82423',

}

framework_change = {
    'SMOTE': 'S.', 'SMOTE-Ge': 'S.-G', 'SMOTE-W': 'S.-W', 'SMOTE-SW': 'S.-SW', 'SMOTE-GB': 'S.-INGB', 'SMOTE_TomekLinks': 'S-TkL',
    'SMOTE_TomekLinks-Ge': 'S-TkL-G', 'SMOTE_TomekLinks-W': 'S-TkL-W', 'SMOTE_TomekLinks-SW': 'S-TkL-SW', 'SMOTE_TomekLinks-GB': 'S-TkL-INGB',
    'SMOTE_ENN': 'S-ENN', 'SMOTE_ENN-Ge': 'S-ENN-G', 'SMOTE_ENN-W': 'S-ENN-W', 'SMOTE_ENN-SW': 'S-ENN-SW', 'SMOTE_ENN-GB': 'S-ENN-INGB',
    'SMOTE_IPF': 'S-IPF', 'SMOTE_IPF-Ge': 'S-IPF-G', 'SMOTE_IPF-W': 'S-IPF-W', 'SMOTE_IPF-SW': 'S-IPF-SW', 'SMOTE_IPF-GB': 'S-IPF-INGB',
}
