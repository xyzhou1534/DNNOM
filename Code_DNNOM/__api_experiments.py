import time
import numpy as np
import random
import pandas as pd
import math
import xlrd,xlwt
from xlutils.copy import copy
import subprocess
import os
import scipy.io
    
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier)
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             f1_score, roc_curve, precision_score, 
                             confusion_matrix,precision_recall_fscore_support)


def now_time_str(colon=True):
    t = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    if not colon:
        t = t.replace(':', '')
    return t


def mean_std(a:list):
    if type(a) != np.array() and type(a) != np.ndarray: a = np.array(a)
    return np.mean(a),np.std(a)


def add_noise(data, noise_rate, random_state=1):
    """
    随机翻转数量为 noise_rate * n 的样本的标签，默认标签只有两类
    :param data: 二分类数据，第一列为标签
    :param noise_rate: 要添加的噪声率
    :param random_state: 随机种子
    :return: None
    """
    random.seed(random_state)
    labels = list(set(data[:, 0]))
    if len(labels) == 1:
        swap_label = {0: 1, 1: 0}
    else:
        swap_label = {labels[0]: labels[1], labels[1]: labels[0]}

    n = data.shape[0]
    noise_num = int(n * noise_rate)
    index = list(range(n))
    random.shuffle(index)

    # data_noise = data[index[:noise_num], :]
    # data_noise[:, 0] = np.array(list(map(lambda x: swap_label[x], 
    #                               data_noise[:, 0])))
    for i in index[:noise_num]:
        data[i, 0] = swap_label[data[i, 0]]


def add_noise_lx(dataset, noise_rate, random_state=1):
    """多分类数据"""
    random.seed(random_state)
    label_cat = sorted(list(set(dataset[:, 0])))  # todo
    new_data = np.array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []  # 记录所有噪声的下标
        n_index = 0
        while True:
            rand_index = int(random.uniform(0, n))  # 每次选择下标
            if rand_index in noise_index_list:  # 如果下标已有，执行下一次while
                continue

            if n_index < noise_num:  # 满足两个条件翻转: 正类且噪声噪声不够
                data[rand_index, 0] = random.choice(other_label)  # todo
                n_index += 1
                noise_index_list.append(rand_index)

            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])
    return new_data


def get_classifier(classifier, random_state=0):
    if classifier == 'BPNN':
        clf = MLPClassifier(random_state=random_state)
    elif classifier == 'KNN':
        clf = KNeighborsClassifier()
    elif classifier == 'SVM':
        # kernel='rbf', gamma='auto',
        clf = SVC(random_state=random_state, probability=True)
    elif classifier == 'DTree':
        clf = DecisionTreeClassifier(random_state=random_state)
    elif classifier == 'LR':
        clf = LogisticRegression(
            random_state=random_state)  # solver='liblinear',
    elif classifier == 'RF':
        clf = RandomForestClassifier(random_state=random_state)
    elif classifier == 'GBDT':
        clf = GradientBoostingClassifier(random_state=random_state)
    elif classifier == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=random_state)
    elif classifier == 'XGBoost':
        clf = XGBClassifier(random_state=random_state)
    elif classifier == 'LightGBM':
        clf = LGBMClassifier(random_state=random_state)
    else:
        assert False, '{} is not exist!'.format(classifier)
    return clf


# def lable_num(y:np.ndarray):
#     count = Counter(y)
#     maj_label, min_label  = max(count, key=count.get), min(count, key=count.get)  
#     mai_num, min_num = max(count.values()), min(count.values())
#     return 


def write_excel(path, value, table_head=None, 
                sheet_name='sheet1', blank_space=False):
    """
    写入表格，若表格不存在，则创建新的表格；若表格存在，则在后面追加新的行
    :param path: 保存路径 (.xls结尾)
    :param value: 值，二维列表
    :param table_head: 首行（表头）
    :param sheet_name:
    :param blank_space:
    :return:
    """
    if not isinstance(value, list):
        value_new = []
        for line in value:
            value_new.append(list(line))
        value = value_new
    try:  # 尝试在已有表格中添加数据
        value_write = value
        if blank_space and value:
            value2 = [['' for _ in range(len(value[0]))]]
            value2.extend(value[1:])
            value_write = value2

        # print(path)
        workbook = xlrd.open_workbook(path)  # 打开工作簿
        sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
        if table_head and worksheet.row_values(0)[:min(len(table_head), 
                            len(worksheet.row_values(0)))] != table_head:
            # 表格第一行与 table_head 不同，则重新写入 table_head
            value2 = [table_head]
            value2.extend(value_write[1:])
            value_write = value2
        rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
        new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
        new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格

        index = len(value_write)  # 获取需要写入数据的行数
        for i in range(0, index):
            for j in range(0, len(value_write[i])):
                # 追加写入数据，从i+rows_old行开始写入
                new_worksheet.write(i + rows_old, j, value_write[i][j])
        new_workbook.save(path)  # 保存工作簿

    except FileNotFoundError:  # 未找到该表格，创建表格
        value_write = [table_head] if table_head else []
        value_write.extend(value[1:])

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet(sheet_name)

        index = len(value_write)
        for i in range(0, index):
            for j in range(0, len(value_write[i])):
                # 以前为什么是  sheet.write(i, j, str(value_write[i][j])) ??
                sheet.write(i, j, value_write[i][j])
        workbook.save(path)



def plot_data_and_balls(data_original, data_sampling, center=(), radius=(), ball_labels=(), color_dark=0.1,
                        title='', x_label='', y_label='', file_path='',
                        image_format=None, show_pic=True, close=True):
    """
    画二维数据点和粒球
    :param data_original:
    :param data_sampling:
    :param center:
    :param radius:
    :param color_dark:  data_original 的颜色， 越接近 0 越浅色
    :param title:
    :param x_label:
    :param y_label:
    :param file_path:
    :param image_format: '.svg', '.eps', '.png', '.jpg'
    :param show_pic:
    :param close:
    :return:
    """
    data0 = data_original[data_original[:, 0] != 1]
    data1 = data_original[data_original[:, 0] == 1]
    c = 1 - color_dark
    plt.plot(data0[:, 1], data0[:, 2], '.', color=(c, c, c), markersize=3)
    plt.plot(data1[:, 1], data1[:, 2], '.', color=(1, c, c), markersize=3)

    data0 = data_sampling[data_sampling[:, 0] != 1]
    data1 = data_sampling[data_sampling[:, 0] == 1]
    # plt.rcParams['figure.figsize'] = (8.0, 8.0)  # 图像大小
    plt.plot(data0[:, 1], data0[:, 2], '.k')
    plt.plot(data1[:, 1], data1[:, 2], '.r')

    if center:
        for i in range(len(center)):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[i][0] + radius[i] * np.cos(theta)
            y = center[i][1] + radius[i] * np.sin(theta)
            color = 'k' if ball_labels[i] == 0 else 'r'
            plt.plot(x, y, color, linewidth=0.4)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis('equal')

    # plt.axis([-2, 3, -1.5, 2])  # make_moons
    # plt.axis([-2, 2, -2, 2])  # make_circles
    # plt.grid(linestyle='-', color='#D3D3D3')  # 网格线

    if show_pic:
        plt.show()
    else:
        if isinstance(image_format, str):
            image_format = [image_format]
        for img in image_format:
            if img in ('.jpg', '.png'):
                plt.savefig(file_path + img, dpi=1000, bbox_inches='tight')
            elif img == '.emf':
                fig = plt.gcf()
                plot_as_emf(fig, filename=file_path + img)
            else:
                plt.savefig(file_path + img)

    if close:
        plt.close()


def plot_hyperplane(clf, X, y,
                    title='', x_label='', y_label='', file_path='', image_format=(), show_pic=True, close=True):
    """
    画分类曲面
    """
    clf.fit(X, y)
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                         np.arange(y_min, y_max, 0.002))
    # predict the point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    # red yellow blue

    plt.contourf(xx, yy, Z, cmap='Pastel1')

    colors = ['k', 'r']
    labels = [0, 1]

    for label in [0, 1]:
        plt.scatter(X[y == labels[label], 0], X[y == labels[label],
                                                1], c=colors[label], s=6, cmap=plt.cm.RdYlBu)

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis('equal')

    # fig = plt.gcf()

    # #  画出支持向量
    #     if draw_sv:
    #         sv = clf.support_vectors_
    #         plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

    if show_pic:
        plt.show()
    else:
        for img in image_format:
            if img in ('.jpg', '.png'):
                plt.savefig(file_path + img, dpi=1000, bbox_inches='tight')
            elif img == '.emf':
                fig = plt.gcf()
                plot_as_emf(fig, filename=file_path + img)
            else:
                plt.savefig(file_path + img)

    if close:
        plt.close()


def plot_as_emf(figure, **kwargs):
    """
    python 画 emf 图片
    http://blog.sciencenet.cn/home.php?mod=space&uid=730445&do=blog&quickforward=1&id=1196366
    """
    inkscape_path = kwargs.get(
        "inkscape", "C:\Program Files\Inkscape\inkscape.exe")
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename + '.svg')
        emf_filepath = os.path.join(path, filename + '.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath,
                         '--export-emf', emf_filepath])
        os.remove(svg_filepath)


def get_metrics(clf, data_train, data_test):
    if len(Counter(data_train[:, 0])) < 2:
        raise ('{} Only one class in the data set {}'.format('  ' * 10, '!' * 10))

    # binary class
    if len(Counter(data_train[:, 0])) == 2:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1]

        accuracy = accuracy_score(data_test[:, 0], predict)  # 准确率
        precision = precision_score(data_test[:, 0], predict)
        recall = recall_score(data_test[:, 0], predict)
        auc = roc_auc_score(data_test[:, 0], predict_proba)  # roc曲线面积
        f1 = f1_score(data_test[:, 0], predict)
        tn, fp, fn, tp = confusion_matrix(
            data_test[:, 0], predict).ravel()  # 混淆矩阵
        g_mean = math.sqrt(recall*(tn/(tn+fp)))
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1,
            'g_mean': g_mean,
        }

    # multi class
    else:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        # predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1]
        predict_proba = clf.predict_proba(data_test[:, 1:])


        accuracy = accuracy_score(data_test[:, 0], predict)
        precision = precision_score(data_test[:, 0], predict, average='macro')
        recall = recall_score(data_test[:, 0], predict, average='macro')
        auc = roc_auc_score(data_test[:, 0],predict_proba,multi_class='ovo',average='macro')  # roc
        f1 = f1_score(data_test[:,0],predict,average='macro')

        p_class, r_class,f_class , support_micro = precision_recall_fscore_support(
            y_true=data_test[:,0], y_pred=predict, beta=0.5, average=None, warn_for=('precision','recall','f-score',)
        )
        macro_gmean = 0
        for i in range(len(p_class)):
            macro_gmean += math.sqrt(p_class[i] *  r_class[i])/len(p_class)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1,
            'g_mean': macro_gmean,
        }

    return metrics


def data_info(data_names: list, noise_rate:float=0):
    ''' 获取数据集信息 '''
    columns = ['data_name', 'samples', 'Features', 'Maj', 'Min','Ir']
    lists = []

    for d, data_name in enumerate(data_names):
        print(data_name)
        data_train, data_test, counters = load_binary_data(
                data_name, noise_rate=noise_rate)
        data = np.vstack((data_train, data_test))
        dicts = Counter(data[:, 0])
        n_maj, n_min = max(dicts.values()), min(dicts.values())
        Ir = round(n_maj/n_min,2)
        lists.append([data_name, data.shape[0],
                      data.shape[1]-1, n_maj, n_min, Ir])

    df = pd.DataFrame(lists, columns=columns, index=None)

    now = now_time_str(colon=False)
    df.to_csv(r'KBsomte/SMOTE/data_infomation/data_info_' + now + '.csv', 
                index=False)


def load_binary_data(data_name, proportion=(0.8, 0.2), noise_rate=0.0,
                    imbalance_rate=0, normalized=True, n_rows=0,
                    random_state=0, add_noise_way='together'):
    """
    最后标签在第一列
    按数据集名称读取2分类数据集
    :param data_name:
    :param proportion: 训练集、测试集比例
    :param noise_rate: 噪声率，只在训练集中添加噪声
    :param imbalance_rate: 按该不平衡比例处理为不平衡数据，默认值0代表不作处理
    :param normalized: 是否归一化
    :param n_rows: 读取行数，若为0，则读取所有行，否则随机读取指定行数
    :param add_noise_way: 'separately' : 正、负类样本分别调价噪声，噪声数均为 少数类样本 * noise_rate,
                          'together' : 正、负类样本合并后一起随机添加噪声
    :return: 训练集、测试集数据，第0列为标签，标签为 0，1
             counters = (samples, features, train0, train1, test0, test1)
    """
    data16 = ['fourclass', 'svmguide1', 'diabetes', 'codrna', 'breastcancer', 'creditApproval', 'votes', 'ijcnn1',
              'svmguide3', 'sonar', 'splice', 'mushrooms', 'clean1', 'madelon_train', 'madelon_test', 'isolet5',
              'isolet1234']
    data_big1 = ['avila', 'letter', 'susy']
    data_big2 = ['mocap', 'poker', 'nomao']
    data_big3 = ['magic', 'skin', 'covtype', 'comedy', 'Online_Retail']
    data_big = data_big1 + data_big2 + data_big3
    uci_extended = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                    'pima', 'segment', 'vehicle', 'wine', 'simulated1', 'simulated2', 'simulated3', 'simulated4',
                    'simulated5', 'simulated6', 'simulated7', 'simulated8', 'simulated9', 'simulated10']  # 不平衡比例不变
    uci_extended_fast = ['breast_tissue2', 'ecoli', 'glass', 'haberman', 'heart', 'iris', 'libra', 'liver_disorders',
                         'pima', 'segment', 'vehicle', 'wine']
    uci_extended_simulated = ['simulated1', 'simulated2', 'simulated3', 'simulated4', 'simulated5', 'simulated6',
                              'simulated7', 'simulated8', 'simulated9', 'simulated10']
    uci_extended_not_exist = ['new-thyroid1', 'new-thyroid2', 'cleveland-0', 'dermatology-6', 'led7digit',
                              'page-blocks0', 'page-blocks-1-3', 'vowel0', 'yeast1', 'yeast2', 'yeast3', 'yeast4',
                              'yeast5', 'yeast-0-2-5-6', 'yeast-0-2-5-7-9', 'yeast-0-3-5-9', 'yeast-0-5-6-7-9',
                              'yeast-1-2-8-9', 'yeast-2_vs_4', 'yeast-2_vs_8', ]
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                  'wine', 'letter', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_small_2_class = ['credit', 'german',
                          'heart1', 'hepatitis', 'horse', 'iono', 'wdbc']
    data_pinjie = ['madelon', 'isolet']
    data_gongye = ['Data_for_UCI_named', 'default of credit card clients', 'diabetic',
                   'Epileptic Seizure Recognition', 'Faults', 'HCV-Egy-Data',
                   'messidor_features', 'OnlineNewsPopularity', 'sat', 'seismic-bumps',
                   'shuttle', 'wilt']

    # KEEL imb_IRlowerThan9
    imb_IRlowerThan9 = ['ecoli-0_vs_1', 'ecoli1', 'ecoli2', 'ecoli3', 'glass0', 'glass-0-1-2-3_vs_4-5-6', 'glass1', 'glass6',
                        'haberman', 'iris0', 'new-thyroid1', 'new-thyroid2', 'page-blocks0', 'pima', 'segment0', 'vehicle0',
                        'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast1', 'yeast3']
    # imb_extended = ['glass0', 'new-thyroid1','yeast1',  'page-blocks0',] # 不平衡比例不变
    imb_extended = ['new-thyroid1', 'page-blocks0', ]

    imb_ir_over_9 = ['abalone-17_vs_7-8-9-10', 'abalone-19_vs_10-11-12-13', 'abalone-20_vs_8-9-10', 'abalone-21_vs_8', 'abalone-3_vs_11', 'abalone19', 'abalone9-18', 'car-good', 'car-vgood', 'cleveland-0_vs_4', 'dermatology-6', 'ecoli-0-1-3-7_vs_2-6', 'ecoli-0-1-4-6_vs_5', 'ecoli-0-1-4-7_vs_2-3-5-6', 'ecoli-0-1-4-7_vs_5-6', 'ecoli-0-1_vs_2-3-5', 'ecoli-0-2-3-4_vs_5', 'ecoli-0-2-6-7_vs_3-5', 'ecoli-0-3-4-6_vs_5', 'ecoli-0-3-4-7_vs_5-6', 'ecoli-0-3-4_vs_5', 'ecoli-0-4-6_vs_5', 'ecoli-0-6-7_vs_3-5', 'ecoli-0-6-7_vs_5', 'ecoli4', 'flare-F', 'glass-0-1-4-6_vs_2', 'glass-0-1-5_vs_2', 'glass-0-1-6_vs_2', 'glass-0-1-6_vs_5', 'glass-0-4_vs_5', 'glass2', 'glass4', 'glass5', 'kddcup-buffer_overflow_vs_back', 'kddcup-guess_passwd_vs_satan', 'kddcup-land_vs_portsweep', 'kddcup-land_vs_satan', 'kddcup-rootkit-imap_vs_back', 'kr-vs-k-one_vs_fifteen', 'kr-vs-k-three_vs_eleven', 'kr-vs-k-zero-one_vs_draw', 'kr-vs-k-zero_vs_eight', 'kr-vs-k-zero_vs_fifteen', 'led7digit-0-2-4-5-6-7-8-9_vs_1', 'lymphography-normal-fibrosis', 'poker-8-9_vs_5', 'poker-8-9_vs_6', 'poker-8_vs_6', 'poker-9_vs_7', 'shuttle-2_vs_5', 'shuttle-6_vs_2-3', 'shuttle-c0-vs-c4', 'shuttle-c2-vs-c4', 'vowel0', 'winequality-red-3_vs_5', 'winequality-red-4', 'winequality-red-8_vs_6-7', 'winequality-red-8_vs_6', 'winequality-white-3-9_vs_5', 'winequality-white-3_vs_7', 'winequality-white-9_vs_4', 'yeast-0-2-5-6_vs_3-7-8-9', 'yeast-0-2-5-7-9_vs_3-6-8', 'yeast-0-3-5-9_vs_7-8', 'yeast-0-5-6-7-9_vs_4', 'yeast-1-2-8-9_vs_7', 'yeast-1-4-5-8_vs_7', 'yeast-1_vs_7', 'yeast-2_vs_4', 'yeast-2_vs_8', 'yeast4', 'yeast5', 'yeast6', 'zoo-3', 'ecoli-0-1_vs_5', 'glass-0-6_vs_5', 'page-blocks-1-3_vs_4']

    assert data_name in uci_extended + data16 + data_big + data_small_2_class + data_pinjie+data_gongye+imb_IRlowerThan9+imb_ir_over_9, 'data set \'{}\' is not exist!'.format(
        data_name)
    assert add_noise_way in ('separately', 'together')

    experiment_path = r'KBsomte'  # TODO: 文件夹 experiment 路径

    if data_name in uci_extended:
        df = pd.read_csv(experiment_path +
                         r'/data_set/uci_extended/' + data_name + '.csv')
        data = df.values
        data = np.hstack((data[:, -1:], data[:, :-1]))
    elif data_name in data_gongye:  # 跑工业数据试试
        df = pd.read_csv(experiment_path +
                         r'/data_set/digital_twin/' + data_name + '.csv')
        data = df.values
    elif data_name in data_big:
        df = pd.read_csv(
            experiment_path + r'/data_set/large_data/' + data_name + '.csv', header=None)
        data = df.values
    elif data_name in data_small:
        df = pd.read_csv(
            experiment_path + r'/data_set/small_data/' + data_name + '.csv', header=None)
        data = df.values
        
    # KEEL imb_IRlowerThan9
    elif data_name in imb_IRlowerThan9:
        # 原始标签在最后一列
        df = pd.read_csv(
            experiment_path + r'/data_set/keel/imb_IRlowerThan9/' + data_name + '.csv', header=None)
        data = df.values
        data = np.hstack((data[:, -1:], data[:, :-1]))  # 标签换到第一列
    elif data_name in imb_ir_over_9:
        # 标签在第一列
        df = pd.read_csv(
            experiment_path + r'/data_set/keel/imb_ir_over_9/' + data_name + '.csv', header=None)
        df =df.iloc[1:,:]
        data = df.values
    else:
        data_mat = scipy.io.loadmat(
            experiment_path + r'/data_set/dataset16/dataset16.mat')
        if data_name == 'madelon':
            data = np.vstack(
                (data_mat['madelon_train'], data_mat['madelon_test']))
        elif data_name == 'isolet':
            data = np.vstack((data_mat['isolet1234'], data_mat['isolet5']))
        else:
            data = data_mat[data_name]

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]

    # 归一化
    data = data.astype(np.float64)
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    # 统计正、负类样本并拆分开
    count = Counter(data[:, 0])
    if len(count) != 2:
        raise Exception('数据集 {} 标签类别数不为2： {}'.format(data_name, count))
    label_maj, label_min = max(count,key=count.get), min(count,key=count.get)
    data_more = data[data[:, 0] == label_maj]
    data_less = data[data[:, 0] == label_min]

    # 标签改为 0，1
    data_more[:, 0] = 0
    data_less[:, 0] = 1
    label_maj, label_min = 0, 1

    # if imbalance_rate != 0 and (data_name not in uci_extended and data_name not in imb_extended):
        # 丢掉部分数据，使数据达到给定不平衡率（uci_extended与imb_extended 中的数据不作处理）
        # assert imbalance_rate > 1
        # print(data_less,type(data_less),data_less.shape)
        # data_less_num = int(data_more.shape[0] // imbalance_rate)
        # print(data_less_num)
        # from _collections_abc import Set as _Set, Sequence as _Sequence
        # print(isinstance(data_less, _Sequence))
        # raise "fuck"
        # data_less = random_sampling(
        #     data_less, data_less_num, random_state=random_state)
        # data_less = random.sample(data_less, k=data_less_num)

    # 分为训练集和测试集
    data_more_train, data_more_test = train_test_split(data_more, 
                                    train_size=0.8, random_state=random_state)
    data_less_train, data_less_test = train_test_split(data_less, 
                                    train_size=0.8, random_state=random_state)

    if add_noise_way == 'separately':
        # 训练集加噪, 多数类少数类都翻转 少数类 * noise_rate
        add_noise(data_more_train, noise_rate=noise_rate * data_less_train.shape[0] / data_more_train.shape[0],
                  random_state=random_state)
        add_noise(data_less_train, noise_rate=noise_rate,
                  random_state=random_state)

    # 合并
    data_train = np.vstack((data_more_train, data_less_train))
    data_test = np.vstack((data_more_test, data_less_test))

    if add_noise_way == 'together':
        add_noise(data_train, noise_rate=noise_rate, random_state=random_state)

    # 分别打乱训练集、测试集
    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # 数据集统计，counters = (samples, features, train0, train1, test0, test1)
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)

    return data_train, data_test, counters


def load_kto2_classes_data_lm(data_name, proportion=(0.8, 0.2), 
        noise_rate=0.0,normalized=True, n_rows=0,
        random_state=0, add_noise_way='together'):
    # dataset k to 2 class
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                  'wine', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_zy = ['abalone', 'balancescale', 'car', 'contraceptive', 'ecoli', 'fourclass', 'frogs', 'glass', 'iris',
               'letter', 'newthyroid', 'nuclear', 'OBS', 'pendigits', 'PhishingData', 'poker', 'satimage', 'seeds',
               'segmentation', 'sensorReadings', 'shuttle', 'svmguide2', 'svmguide4', 'userknowledge', 'vehicle',
               'vertebralColumn', 'vowel', 'wifiLocalization', 'yeast', 'krkopt', 'shuttle_all',
               'Healthy_Older_People2']

    data_all = data_small + data_zy

    if len(set(data_all)) != len(data_all):  # 以上数据集列表中有重复的数据集名称
        d = filter(lambda x: x[1] != 1, Counter(data_all).items())
        raise Exception('数据集重复：{}'.format(list(d)[0][0]))

    experiment_path = r'KBsomte/'  # todo: 文件夹 experiment 路径

    if data_name in data_small:
        df = pd.read_csv(experiment_path +
                         r'data_set/small_data/' + data_name + '.csv', header=None)
        # df = pd.read_csv(
        #     r'/home/zhouhao/GITHUB_res/SMOTE大实验/KBsomte/data_set/small_data/' + data_name + '.csv', header=None)
        data = df.values
    elif data_name in data_zy:
        df = pd.read_csv(experiment_path +
                         r'data_set/DataSet/' + data_name + '.csv', header=None)
        # df = pd.read_csv(
        # r'/home/zhouhao/GITHUB_res/SMOTE大实验/KBsomte/data_set/DataSet/' + data_name + '.csv', header=None)
        data = df.values
    else:
        assert False, 'data set \'{}\' is not exist!'.format(data_name)

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]

    # 归一化
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    # 统计正、负类样本并拆分开
    count = Counter(data[:, 0])
    # if len(count) == 2:
    #     raise Exception('数据集 {} 标签类别数为2： {}'.format(data_name, count))
    # print(len(count))
    count_dict = dict(count)    # 将counter类型转换为字典
    # print(type(count_dict), count_dict)
    count_sorted_values = sorted(
        count_dict.items(), key=lambda x: x[1], reverse=True)  # 根据数据量排序
    # print(count_sorted_values)
    # print(count_sorted_values[-1][0], count_sorted_values[0][0])
    # 获取少数类与多数类标签
    tp_less, tp_more = count_sorted_values[-1][0], count_sorted_values[0][0]
    data_more = data[data[:, 0] != tp_less]
    data_less = data[data[:, 0] == tp_less]
    # print(len(data_more), len(data_less))

    # 标签改为 0，1
    data_more[:, 0] = 0
    data_less[:, 0] = 1
    tp_more, tp_less = 0, 1
    # 分为训练集和测试集
    data_more_train, data_more_test = train_test_split(
                        data_more, train_size=0.8, random_state=random_state)
    data_less_train, data_less_test = train_test_split(
                        data_less, train_size=0.8, random_state=random_state)

    if add_noise_way == 'separately':
        # 训练集加噪, 多数类少数类都翻转 少数类 * noise_rate
        add_noise(data_more_train, noise_rate=noise_rate * data_less_train.shape[0] / data_more_train.shape[0],
                  random_state=random_state)
        add_noise(data_less_train, noise_rate=noise_rate,
                  random_state=random_state)

    # 合并
    data_train = np.vstack((data_more_train, data_less_train))
    data_test = np.vstack((data_more_test, data_less_test))

    if add_noise_way == 'together':
        add_noise(data_train, noise_rate=noise_rate, random_state=random_state)

    # 分别打乱训练集、测试集
    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # 数据集统计，counters = (samples, features, train0, train1, test0, test1)
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)

    return data_train, data_test, counters



def load_k_classes_data(data_name, proportion=(0.8, 0.2), noise_rate=0.0, normalized=True, random_state=0, n_rows=0):
    data_small = ['anneal', 'credit', 'german', 'heart1', 'heart2', 'hepatitis', 'horse', 'iono', 'sonar', 'wdbc',
                'wine', 'lymphography', 'mushroom', 'soybean', 'zoo']
    data_zy = ['abalone', 'balancescale', 'car', 'contraceptive', 'ecoli', 'fourclass', 'frogs', 'glass', 'iris',
            'letter', 'newthyroid', 'nuclear', 'OBS', 'pendigits', 'PhishingData', 'poker', 'satimage', 'seeds',
            'segmentation', 'sensorReadings', 'shuttle', 'svmguide2', 'svmguide4', 'userknowledge', 'vehicle',
            'vertebralColumn', 'vowel', 'wifiLocalization', 'yeast', 
            'krkopt','shuttle_all','Healthy_Older_People2']

    data_all = data_small + data_zy
    experiment_path = r'KBsomte/'   # 相对路径

    if len(set(data_all)) != len(data_all):  # 以上数据集列表中有重复的数据集名称
        d = filter(lambda x: x[1] != 1, Counter(data_all).items())
        raise Exception('数据集重复：{}'.format(list(d)[0][0]))

    if data_name in data_small:
        df = pd.read_csv(experiment_path+
                         r'data_set/small_data/'  + data_name + '.csv', header=None)
        data = df.values
    elif data_name in data_zy:
        df = pd.read_csv(experiment_path + r'data_set/DataSet/'
            + data_name + '.csv', header=None)
        data = df.values
    else:
        assert 0 == 1, 'data set \'{}\' is not exist!'.format(data_name)

    np.random.seed(random_state + 4)
    np.random.shuffle(data)
    if 0 < n_rows < data.shape[0]:
        data = data[:n_rows]

    # 归一化
    if normalized:
        for f in range(1, data.shape[1]):
            ma, mi = max(data[:, f]), min(data[:, f])
            subtract = ma - mi
            if subtract != 0:
                data[:, f] = (data[:, f] - mi) / subtract

    count = Counter(data[:, 0])
    data_train = np.array([]).reshape((-1, data.shape[1]))
    data_test = np.array([]).reshape((-1, data.shape[1]))
    labels = list(set(data[:, 0]))
    for label in labels:
        train, test = train_test_split(data[data[:, 0] == label, :],
                                    train_size=0.8, random_state=random_state)
        data_train = np.vstack((data_train, train))
        data_test = np.vstack((data_test, test))

    data_train = add_noise_lx(
        data_train, noise_rate=noise_rate, random_state=random_state)

    np.random.seed(random_state + 4)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    # 数据集统计，counters = (samples, features, train0, train1, test0, test1)
    counters = (data_train.shape[0] + data_test.shape[0], data_train.shape[1] - 1, Counter(data_train[:, 0])[0],
                Counter(data_train[:, 0])[1], Counter(data_test[:, 0])[0], Counter(data_test[:, 0])[1], count)

    return data_train, data_test , counters