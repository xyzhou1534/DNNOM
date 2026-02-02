
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

framework_methods = ['SMOTE-GB', 'SMOTE', 'SMOTE-G', 'SMOTE-W', 'SMOTE-SW',
                     'SMOTE_ENN-GB', 'SMOTE_ENN', 'SMOTE_ENN-G', 'SMOTE_ENN-W', 'SMOTE_ENN-SW',
                     'SMOTE_TomekLinks-GB', 'SMOTE_TomekLinks', 'SMOTE_TomekLinks-G', 'SMOTE_TomekLinks-W', 'SMOTE_TomekLinks-SW',
                     'SMOTE_IPF-GB', 'SMOTE_IPF', 'SMOTE_IPF-G', 'SMOTE_IPF-W', 'SMOTE_IPF-SW', ]
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
    'new-thyroid1': 'n-thy1',
    'ecoli': 'ecoli',
    'wisconsin': 'wiscon',
    'diabetes': 'diabet',
    'breastcancer': 'breast',
    'vehicle2': 'vehic2',
    'vehicle': 'vehicl',
    # 'nuclear': 'nuclear',
    'yeast1': 'yeast1',
    'Faults': 'Faults',
    'segment': 'segment',
    # 'seismic-bumps': 'seismi',
    'satimage': 'satimage',
    'wilt': 'wilt',
    'svmguide1': 'svmguide1',
    'mushrooms': 'mushro',
    # 'sensorReadings': 'sReadi',
    'page-blocks0': 'page-b',
    'Data_for_UCI_named': 'UCI-na',
    # 'frogs': 'frogs',
    'letter': 'letter',
    'avila': 'avila',
    'magic': 'magic',
    'susy': 'susy',
    'default of credit card clients': 'dccc',
    'nomao': 'nomao',
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
