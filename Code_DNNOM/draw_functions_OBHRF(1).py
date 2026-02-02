from api_GB import (framework_methods_spe, vsothers, Metrics, Classifiers, Noise_rates, others_wilcoxon, framework_wilcoxon, Metrics_dict, framework_color_dict, framework_change, framework_multi, vsothers_multi, data_name_multi)
from api_OBHRF import framework_methods, data_name, method_name

import time
import pandas as pd
from cProfile import label
# from tkinter.tix import Tree
import warnings
import math

from sympy import rotations
warnings.filterwarnings('ignore')


Classifiers = ['AdaBoost', 'DTree', 'GBDT', 'KNN', 'LightGBM', 'LR', 'SVM', 'XGBoost']
Metrics = ['precision', 'recall', 'f1', 'g_mean', 'AUC']


def table_OBHRF1(filepath: str, is_save=0,noise_rate=0):
    """
        整理所有噪声下，所有数据集在每个分类器上的结果。（所有噪声率下的平均）
    """
    
    # 读取excel
    data = pd.read_excel(filepath, sheet_name='sheet1')
    # print('data:', type(data),data.shape)
    # 筛选噪声率
    # data = data.loc[data['noise rate'] == noise_rate]
    # print('data:', type(data),data.shape)

    # make table
    table_head = []
    

    for classif in Classifiers:
        for metric in Metrics:
            table_head.append([metric])
            metric_temp = []
            for dataset in data_name.keys():
                dataset_res = [data_name[dataset]]    # 保存每个数据集的结果
                dataset_var = [data_name[dataset]]
                for method in framework_methods:
                    # print('当前工作：\t', metric, dataset, method)
                    data_t = data.loc[data['Classifier'] == classif]
                    # print(data_t)
                    data_t = data_t.loc[data_t['data'] == dataset]
                    # print(data_t)
                    data_t = data_t.loc[data_t['sampling method']
                                        == method, metric]
                    # print(data_t.shape)
                    # print(data_t)
                    
                    temp_res = data_t.mean()     # 平均结果
                    temp_dif = data_t.std()      # 方差

                    dataset_res.append(str(format(temp_res, '.3f')))
                    dataset_var.append(f'{format(temp_res, ".3f")}±{format(temp_dif, ".3f")}')
                    # print('res: ', temp_res, '    dif: ', temp_dif)

                # # 'xxx'表示加粗
                # for i in range(2, len(dataset_res), 2):
                #     if float(dataset_res[i].split()[0]) >= float(dataset_res[i-1].split()[0]):
                #         dataset_res[i] = 'XXX ' + dataset_res[i]
                #         dataset_var[i] = 'XXX ' + dataset_var[i]
                #     else:
                #         dataset_res[i-1] = 'XXX ' + dataset_res[i-1]
                #         dataset_var[i-1] = 'XXX ' + dataset_var[i-1]

                metric_temp.append(dataset_var)

            # 计算 Average
            Average = ['Average']
            l_dn = len(data_name)
            for col in range(1, len(metric_temp[0]), 1):
                res, std = 0, 0
                for row in range(l_dn):
                    r, s = metric_temp[row][col].split('±')
                    res += float(r)
                    std += float(s)
                
                Average.append(f'{str(format(res/l_dn, ".3f"))}±{str(format(std/l_dn, ".3f"))}')
            
            # 计算win_times
            
            Winning_times = ['Winning times']
            cnt = 0
            for col in range(1, len(metric_temp[0]), 2):
                Winning_times.append('')  # 占一列, 只在第二列显示比例
                win_times = 0
                for row in range(l_dn):
                    r1, s1 = metric_temp[row][col].split('±')  # 原算法
                    r2, s2 = metric_temp[row][col+1].split('±')  # 加上框架
                    if r1 < r2:  # 框架好 
                        win_times += 1
                Winning_times.append(f'{str(l_dn-win_times)}: {str(win_times)}')
            
            metric_temp.append(Average)
            metric_temp.append(Winning_times)
            table_head.extend(metric_temp)
            
            
    # 运行完毕, table_head是整个excel表, 嵌套列表[[], [], [], ...]
    table_res = []
    l_f = len(framework_methods) // 2
    for method in [method_name[framework_methods[i]] for i in range(0, l_f*2, 2)]:  # 15行, 第一二列的元素
        table_res.append([str(noise_rate), method]) 
    
    l_m = len(Metrics)  # 指标个数
    l_c = len(Classifiers)  # 分类器个数
    cnt1 = 0
    temp_sum = [[[0, 0] for _ in range(l_m*2)] for _ in range(l_f)]  # 15行 10列, 数据部分
    for i in range(len(table_head)):  # 遍历整个表格
        if table_head[i][0] == 'Average':  # 找到平均值的一行, 全部填入
            for j in range(1, len(table_head[i]), 2):  # 一行有30个元素 == 算法个数*2
                r1 = table_head[i][j].split('±')  # origin  [v, std]
                r2 = table_head[i][j+1].split('±')  # DNNOM
                # 竖着添加, 每一行Average添加为temp_sum的一列, 前5行把temp_sum填满
                k = cnt1%l_m*2  # 列数[0, 2, 4, 8]
                temp_sum[j//2][k][0] += float(r1[0])  # 平均值
                temp_sum[j//2][k][1] += float(r1[1])  # 方差
                temp_sum[j//2][k+1][0] += float(r2[0])
                temp_sum[j//2][k+1][1] += float(r2[1])
            cnt1 += 1
    # print('temp_sum: ')
    # print(temp_sum)
    # exit()
    
    # 上面完成了全部的求和, temp_sum 是表格的所有值, 下面规范temp_sum, 并累加
    Winning_times = [str(noise_rate), 'Win-times']
    ave = [[0, 0] for _ in range(l_m*2)]
    win_times = [0 for _ in range(l_m*2)]
    for i in range(l_f):
        for j in range(0, len(temp_sum[0]), 2):  # 一次处理两列, 方便对比
            v1 = temp_sum[i][j][0] / l_c
            std1 = temp_sum[i][j][1] / l_c
            v2 = temp_sum[i][j+1][0] / l_c
            std2 = temp_sum[i][j+1][1] / l_c
            ave[j][0] += v1
            ave[j][1] += std1
            ave[j+1][0] += v2
            ave[j+1][1] += std2
            temp_sum[i][j] = f'{str(format(v1, ".3f"))}±{str(format(std1, ".3f"))}'
            temp_sum[i][j+1] = f'{str(format(v2, ".3f"))}±{str(format(std2, ".3f"))}'
            
            if v2 > v1:  # DNNOM win
                win_times[j+1] += 1
            else: win_times[j] += 1
    
    for i in range(l_f):  # 合并
        temp_sum[i].append('')
        table_res[i].extend(temp_sum[i])
    
    # 求取平均值
    Average = [str(noise_rate), 'Average']
    for i in range(len(ave)):  # 
        ave[i][0] /= l_f
        ave[i][1] /= l_f
        Average.append(f'{str(format(ave[i][0], ".3f"))}±{str(format(ave[i][1], ".3f"))}')
    
    Average.append('')
    table_res.append(Average)  # 合并
    win_times.append('')
    Winning_times.extend(win_times)
    table_res.append(Winning_times)
    
    
    if is_save:
        table_header = ['datasets', 
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
                        'SMOTE_RandomUnderSampler', 'SMOTEN_RandomUnderSampler_OIRP_BH', 
                        'SMOTEN_ClusterCentroids', 'SMOTEN_ClusterCentroids_OIRP_BH', 
                        'SMOTEN_NearMiss', 'SMOTEN_NearMiss_OIRP_BH', 
                        'SMOTEN_RandomUnderSampler', 'SMOTEN_RandomUnderSampler_OIRP_BH', 
                        'SVMSMOTE_ClusterCentroids', 'SVMSMOTE_ClusterCentroids_OIRP_BH', 
                        'SVMSMOTE_NearMiss', 'SVMSMOTE_NearMiss_OIRP_BH', 
                        'SVMSMOTE_RandomUnderSampler', 'SVMSMOTE_RandomUnderSampler_OIRP_BH']
        
        table_header1 = ['nr', 'Oversampler', 'original', 'OIRP', 'original', 'OIRP', 'original', 'OIRP', 'original', 'OIRP', 
                         'original', 'OIRP', 'original/OIRP']
        
        table_data = pd.DataFrame(table_head)
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        table_data.to_csv(r'./DNNOM/res/' +
                        f'table{str(now)}_{str(noise_rate)}'+'.csv', index=False,  header=table_header) 
        print('成功生成表格:', f'table{str(now)}_{str(noise_rate)}'+'.csv')
        
        table_data = pd.DataFrame(table_res)
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        table_data.to_csv(r'./DNNOM/res/' +
                        f'res{str(now)}_{str(noise_rate)}'+'.csv', index=False,  header=table_header1) 
        print('成功生成结果:', f'res{str(now)}_{str(noise_rate)}'+'.csv')

if __name__ == '__main__':

    # ============================== OBHRF ===================================
    # 整理指定噪声下，所有数据集在单个分类器上的结果
    # ph = [r'OBHRF/data5.xlsx', r'OBHRF/data15.xlsx', r'OBHRF/data25.xlsx', r'OBHRF/data35.xlsx', 
    #       r'OBHRF/data45.xlsx']
    
    # for i in range(1, 10, 2):
    #     table_OBHRF1(filepath=ph[i//2], is_save=1, noise_rate=0.05*i)
        
        
    # filepath = r'OBHRF/data15.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.15)
    
    # filepath = r'OBHRF/data25.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.25)
    
    # filepath = r'OBHRF/data35.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.35)
    
    filepath = r'OBHRF/data45.xls'
    
    table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.45)
    
    # filepath = r'OBHRF/data15.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.15)
    
    # filepath = r'OBHRF/data25.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.25)
    
    # filepath = r'OBHRF/data35.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.35)
    
    # filepath = r'OBHRF/data45.xls'
    
    # table_OBHRF1(filepath=filepath, is_save=1, noise_rate=0.45)