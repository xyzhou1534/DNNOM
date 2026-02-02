import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 全局变量
Metric = ['Precision', 'Recall','AUC', 'F1', 'G-mean' ]

# nr = [0.05, 0.15, 0.25, 0.35, 0.45]
# nr = [0.15, 0.25]
nr = [0.35, 0.45]

Classifier = ['AdaBoost', 'DTree', 'GBDT', 'KNN', 'LightGBM', 'LR', 'SVM', 'XGBoost']

samplers = [
                'BorderlineSMOTE_ClusterCentroids', 'BorderlineSMOTE_ClusterCentroids_OIRP_BH', 
                'BorderlineSMOTE_NearMiss', 'BorderlineSMOTE_NearMiss_OIRP_BH', 
                'BorderlineSMOTE_RandomUnderSampler', 'BorderlineSMOTE_RandomUnderSampler_OIRP_BH', 
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
                'SVMSMOTE_RandomUnderSampler', 'SVMSMOTE_RandomUnderSampler_OIRP_BH',
                ]

xname = ['Bls.Cc.', 'Bls.Cc._D', 
         'BlS.Nm.', 'BlS.Nm._D', 
         'BlS.Rus.', 'BlS.Rus._D', 
         'Ros.Cc.', 'Ros.Cc._D', 
          'Ros.Nm.', 'Ros.Nm._D', 
          'Ros.Rus.', 'Ros.Rus._D', 
          'S.Cc.', 'S.Cc._D', 
          'S.Nm.', 'S.Nm._D', 
          'S.Rus.', 'S.Rus._D', 
          'SN.Cc.', 'SN.Cc._D', 
          'SN.Nm.', 'SN.Nm._D', 
          'SN.Rus.', 'SN.Rus._D', 
          'SvS.Cc.', 'SvS.Cc._D', 
          'SvS.Nm.', 'SvS.Nm._D', 
          'SvS.Rus.', 'SvS.Rus._D',
          ]


# 构建升序排序矩阵
def rank_matrix(matrix):
    '''
    description: called by main()
    param {*}
    return {*}
    '''
    # print('原始矩阵：\n',matrix)
    rnum = matrix.shape[0]  # 行
    cnum = matrix.shape[1]  # 列
    sorts = np.argsort(-matrix)  # 按降序序返回索引,返回从大到小的元素的索引

    for i in range(rnum):
        k = 1  # 相同的个数
        n = 1
        flag = False
        nsum = 0  # 排序和

        for j in range(cnum):  # 每列
            n = n + 1

            # 找到每行相等的
            if j < 7 and matrix[i, sorts[i, j]] == matrix[i, sorts[i, j + 1]]:  # TODO 算法个数

                flag = True
                k = k + 1  # 相同的个数
                nsum += j + 1  # 排序和

            # 为相等的赋值
            elif (j == 7 or
                  (j < 7 and
                   (matrix[i, sorts[i, j]] == matrix[i, sorts[i, j + 1]] or
                    matrix[i, sorts[i, j]] == matrix[i, sorts[i, j - 1]])))\
                    and flag:

                nsum += j + 1
                for q in range(k):  # 共有k个列相同，为每个列赋值
                    matrix[i, sorts[i, j - k + q + 1]] = nsum / k  # 排序和/个数
                k = 1
                flag = False
                nsum = 0

            # 为不相等的赋值
            else:
                matrix[i, sorts[i, j]] = j + 1  # 第几大的就赋值几
                # continue
        # print('本行循环结束！：\n',matrix[i],'\n')

    # print('\n最后的矩阵：\n',matrix)
    return matrix

def friedman(n, k, rank_matrix):
    # rank_matrix: 表示第i个算法的平均序值
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    # print(sumr)
    result = 12 * n / (k * (k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result / (n * (k - 1) - result)
    return result

def plt_bar2(N, values, metric, classifier, nr, axes, t, j):
    width = 0.4     # 柱子的宽度
    l_v = len(values)
    color = {   # 选用配色
             0: '#fdcabf', 2: "#bbcfe8",  # 15%
             1: "#a30543", 3: '#234086',  # 25%
             
             4: "#efc99b", 6: '#e5cbe1',  # 35%
             5: "#e19d49", 7: '#3c3b8b',  # 45%
             
             
             8: "#b3cede", 9: "#225b91", 
             }
    # 选用字体
    font1 = {
        'size': 35,
        'family': 'Times New Roman',
        'rotation': 90,
    }
    font2 = {
        'size': 100,
        'family': 'Times New Roman',
        'rotation': 0,
    }
    font3 = {
        'size': 50,
        'family': 'Times New Roman',
        'rotation': 0,
    }
    font4 = {
        'size': 70,
        'family': 'Times New Roman',
        'rotation': 90,
    }
    
    x = np.arange(l_v)
    ind = [i * width for i in range(len(x)//2)for _ in range(2)]
    ind = np.array(ind)
    x_tem = ind+(x*0.4)
    for i in range(2, len(x_tem), 2):
        x_tem[i] -= i//2 * 0.3
        x_tem[i+1] -= i//2 * 0.3
    x_data_0 = [x_tem[i] for i in range(0, l_v, 2)]
    x_data_1 = [x_tem[i] for i in range(1, l_v, 2)]
   
    
    # 不同噪声率对应不同柱子颜色
    if nr == 0.05:
        axes.bar(x_data_0, [l_v-values[i] for i in range(0, l_v, 2)], width, color=color[0], label='origin')
        axes.bar(x_data_1, [l_v-values[i] for i in range(1, l_v, 2)], width, color=color[2], label='DNNOM')
    elif nr == 0.15:
        axes.bar(x_data_0, [l_v-values[i] for i in range(0, l_v, 2)], width, color=color[0], label='origin')
        axes.bar(x_data_1, [l_v-values[i] for i in range(1, l_v, 2)], width, color=color[2], label='DNNOM')
    elif nr == 0.25:
        axes.bar(x_data_0, [l_v-values[i] for i in range(0, l_v, 2)], width/2, color=color[1], label='origin')
        axes.bar(x_data_1, [l_v-values[i] for i in range(1, l_v, 2)], width/2, color=color[3], label='DNNOM')
    elif nr == 0.35:
        axes.bar(x_data_0, [l_v-values[i] for i in range(0, l_v, 2)], width, color=color[4], label='origin')
        axes.bar(x_data_1, [l_v-values[i] for i in range(1, l_v, 2)], width, color=color[6], label='DNNOM')
    elif nr == 0.45:
        axes.bar(x_data_0, [l_v-values[i] for i in range(0, l_v, 2)], width/2, color=color[5], label='origin')
        axes.bar(x_data_1, [l_v-values[i] for i in range(1, l_v, 2)], width/2, color=color[7], label='DNNOM')
    



    if j == 0:  # 第1列设置分类器名字和刻度
        axes.set_ylabel(classifier+'\nMean Rank', fontdict=font4)
        axes.set_ylim(8, 23)
        axes.set_yticks([i for i in range(8, 24, 3)])
        axes.set_yticklabels([23, 20, 17, 14, 11, 8], fontdict=font3)

    if t != 1:
        axes.set_xticks([])  # 隐藏横坐标刻度
    if t == 0:  # 第1行设置指标
        axes.set_title(metric, font2)

    if t == 7:  # 第8行设置90度的X轴标签
        axes.set_xticks(x_tem)
        axes.set_xticklabels(xname, fontdict=font1)
    
    if t == 7 and j == 0: # 假设你有3列
        axes.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 25})
        
    
        
    


def main(is_save:bool, datasets: int, sampling_method: int)->None:
    # ph = [r'./DNNOM/data5.xls', r'./DNNOM/data15.xls', r'./DNNOM/data25.xls', r'./DNNOM/data35.xls', r'./DNNOM/data45.xls', ]
    ph = [
        #   r'./DNNOM/data5.xls', 
        #   r'./DNNOM/data15.xls', 
        #   r'./DNNOM/data25.xls', 
          r'./DNNOM/data35.xls', 
          r'./DNNOM/data45.xls', 
          ]

    # 读取全部表格并合并
    df = pd.concat([pd.read_excel(p, sheet_name='sheet1') for p in ph], ignore_index=True)
    df = pd.DataFrame(df)
    
    res = []
    res_mean = []

    classifier = ''
    metric = ''
    noise_rate = ''
    rk_matrix_mean = ''
    samplers_k = ''
    n = 1
    fig, axes = plt.subplots(len(Classifier), len(Metric), figsize=(
        100, 60), sharey=True, sharex=True, squeeze=False)

    

    for i in range(len(nr)):
        # 每个noise是一幅大图
        # fig, axes = plt.subplots(5, 4, figsize=(
        #     50, 50), sharey=True, sharex=True)
        for t in range(len(Metric)):  # 5列
            metric = Metric[t]
            for j in range(len(Classifier)):  # 8行
                classifier = Classifier[j]
                matrix_dic = {}
                matrix_dic[Metric[t]] = pd.DataFrame()  # 字典的值是dataframe
                for k in range(len(samplers)):
                    df_0 = df.loc[df['noise rate'] == nr[i]]
                    df_0 = df_0.loc[df['Classifier'] == classifier]
                    df_0 = df_0.loc[df_0['sampling method'] == samplers[k]]
                    df_0 = df_0[Metric[t]]
                    df_0 = df_0.reset_index(drop=True)  # 重置索引，删除索引
                    matrix_dic[Metric[t]] = pd.concat(
                        [matrix_dic[Metric[t]], df_0], axis=1)

                matrix_t = matrix_dic[Metric[t]].values  # 字典的值转矩阵
                rk_matrix = rank_matrix(matrix_t)  # 把指标矩阵转为秩矩阵
                
                rk_matrix_mean = (rk_matrix.mean(axis=0))  # 对列求均值,用均值来画图
                
                # friedman验证结果
                friedman_test = friedman(datasets, sampling_method, rk_matrix)    # 17个数据集 + 15个方法

                s = str(nr[i]) + '_' + metric
                # print('当前工作：\t', n, ',', s, rk_matrix_mean, '\n')
                res.append([s, abs(friedman_test)])  # 保留最后结果到列表，取绝对值
                
                res_mean.append(
                    [noise_rate, classifier, metric] + rk_matrix_mean.tolist())

                metric = Metric[t]
                noise_rate = nr[i]
                
                print('当前工作', '\nmetric: ', metric, '\nclassifier: ', classifier, '\nnr: ', noise_rate, '\n')
                
                # plt_bar2(8, rk_matrix_mean, metric, classifier, noise_rate, axes[j, t], j, t)
                n += 1
        
    # lines, labels = fig.axes[0].get_legend_handles_labels()
    # print(lines, labels)
    # fig.legend(lines, labels, loc='lower center', ncol=2, prop={'family': 'Times New Roman', 'size': 40})
    # print(1)

    # if is_save:
    #     now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    #     plt.tight_layout()
    #     plt.savefig(r'./DNNOM/'+str(noise_rate) +'_'+now+'_.pdf',format='pdf',dpi=600)  # TODO: 保存图片路径
    #     plt.close()

    # res = pd.DataFrame(res)
    # res.to_excel(r'./DNNOM/res/'+now_time_str(colon=False).strip('[]') +
    #              '.xlsx', index=False)  # 整体结果保留位xls文件

    res_mean = pd.DataFrame(res_mean)
    res_mean.to_excel(r'./DNNOM/res/mean/' +
                      now_time_str(colon=False).strip('[]')+'.xlsx', index=False)


def now_time_str(colon=True):
    t = str(time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time())))
    if not colon:
        t = t.replace(':', '')
    return t


if __name__ == '__main__':

    # stac_zhou()
    main(is_save=1, datasets=17, sampling_method=15)
    print('程序运行完毕！')
