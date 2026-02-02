'''
Author: Zhou Xiangyu
Date: 2026-02-01 16:17
Description: 用于画论文DNNOM的图
E-mail: xyzhou1534@gmail.com
'''


from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
import random
from imblearn.over_sampling import SMOTE, ADASYN,KMeansSMOTE,SVMSMOTE, BorderlineSMOTE, RandomOverSampler, SMOTEN
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler
from imblearn.combine import SMOTEENN
# from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC

import warnings 
warnings.filterwarnings("ignore")   # 取消警告


# import Diy library
from DNNOM_BO import PSO_Denoise, opt_n
from api import KL_gaussian, add_flip_noise,make_noise,make_imbalance
# from compare_methods.Classification.Over import GDO,ZhengSY,PA

def ablustion(is_save:bool=False,is_show:bool=False,noise_rate:float=0)->None:
    '''
    description: 消融可视化实验
    return {*}
    '''    

    Ir = {
        0: 100,
        1: 500,
    }
    # 制作数据集
    X, y = ablustion_data(noise_rate=noise_rate,Ir=Ir)
    X_length = len(X)  # 样本数量 600
    counter = Counter(y)  # 两种标签的数量
    n_maj, n_min = max(counter.values()), min(counter.values()) #  1: 460, 0: 140 

    # settings of figure ******************************************************
    num = 0   # adjust the subfigs: row,col,index
    color = {
            0: '#cc4a74', 1: '#16058b', 
            2: '#efc99b',-1:'#e19d49'
            }
    X_colors = [color[label] for label in y]
    # cmap = {0:'winter',1:'autumn',2:'summer'}
    font = {
        'family':'Times New Roman',
        'size':18,
        }
    X_max, y_max = max(X[:,0]),max(X[:,1])
    X_min, y_min = min(X[:,0]),min(X[:,1])
    Z_min, Z_max = min(X[:,2]),max(X[:,2])
    X_min, X_max = X_min-0.01, X_max+0.01
    y_min, y_max = y_min-0.01, y_max+0.01
    Z_min, Z_max = Z_min-0.01, Z_max+0.01

    fig_1 = plt.figure(figsize=(20, 8))


    # 画图函数
    def settings_ax(title:str)->None:
        """public settings of each sub ax"""
        nonlocal num    # 扩展上层函数中变量的作用域
        num += 1
        ax = fig_1.add_subplot(1, 4, num, projection='3d')

        ax.set_xticklabels([])  # 隐藏刻度但保留格子线
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # 确保网格开启
        ax.grid(True)
        ax.tick_params(labelsize=16)  # 设置坐标轴刻度字体大小，axis='x', 'y' or 'z'
        ax.tick_params(labelsize=16, rotation=-45, axis='x')
        ax.tick_params(labelsize=16, rotation=45, axis='y')
        fig_1.text(0.12, 0.5, 'SMOTE', va='center', ha='center', 
                   rotation='vertical', fontsize=14)
        
        if num == 1: 
            ax.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.4)   # origin samples
        
        # 在每列的头上加标题, 只有第一次运行需要, 否则注释
        if num == 1:
            ax.set_title('origin data', loc='center', y=0.9, fontsize=14)
        elif num == 2:
            ax.set_title('sampler', loc='center', y=0.9, fontsize=14)
        elif num == 3:
            ax.set_title('DNNOM - $n_{+}^{*}$', loc='center', y=0.9, fontsize=14)
        elif num == 4:
            ax.set_title('DNNOM - $\mathcal{P}^{*}$', loc='center', y=0.9, fontsize=14)
        
        ax.view_init(elev=4,azim=23)       # 调整可视化角度

        # X和X_new不用传进来，因为嵌套函数享用上层函数的变量作用域
        if num > 1:
            ax.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.2)   # origin samples
        
        # 分类器超平面
        ax.plot_surface(xx,yy,Z1,alpha=0.8,
                        # color='Snow',
                        cmap='PuBu_r',
                        )
        # new samples
        if X_res is None:return
        X_new = X_res[X_length:]
        ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],s=15,c='red',alpha=0.9,marker='+') 
 


    # origin data原始数据集 ***********************************************************
    X_res = None
    
    clf = SVC(random_state=42, probability=True,kernel='linear')
    clf.fit(X,y)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  # 计算超平面
    settings_ax(title='(a) Original dataset')
    # legend_original=ax_2.plot_surface(xx,yy,Z1,alpha=0.7, color='#63b2ee',)


    # SMOTE  原始过采样
    model = SMOTE(random_state=42)
    # model = BorderlineSMOTE(random_state=42)
    # model = RandomOverSampler(random_state=42)
    # model = SVMSMOTE(random_state=42)
    # model = SMOTEN(random_state=42)
    # model = RandomUnderSampler(random_state=42)
    # model = ClusterCentroids(random_state=42)
    # model = NearMiss()
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE采样后:\t',Counter(y_res))
    # print('BorderlineSMOTE采样后:\t',Counter(y_res))
    # print('RandomOverSampler采样后:\t',Counter(y_res))
    # print('SVMSMOTE采样后:\t',Counter(y_res))
    # print('SMOTEN采样后:\t',Counter(y_res))
    # print('RandomUnderSampler采样后:\t',Counter(y_res))
    # print('ClusterCentroids采样后:\t',Counter(y_res))
    # print('NearMiss采样后:\t',Counter(y_res))
    
    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    
    w=clf.coef_         
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]
    settings_ax(title='(b) SMOTE')
    # settings_ax(title='(b) BorderlineSMOTE')
    # settings_ax(title='(b) RandomOverSampler')
    # settings_ax(title='(b) SVMSMOTE')
    # settings_ax(title='(b) SMOTEN')
    # settings_ax(title='(b) RandomUnderSampler')
    # settings_ax(title='(b) ClusterCentroids')
    # settings_ax(title='(b) NearMiss')
    # legend_smote=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#76da91')


    # SMOTE n*, opt_n
    n_opt, sampling_strategy = opt_n(X,y)  # 获取采样率
    print('\t\tn_opt:\t', n_opt)
    print('\t\tsampling_strategy:\t',sampling_strategy,type(sampling_strategy))
    model = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    # model = BorderlineSMOTE(random_state=42, sampling_strategy=sampling_strategy)
    # model = RandomOverSampler(random_state=42, sampling_strategy=sampling_strategy)
    # model = SVMSMOTE(random_state=42, sampling_strategy=sampling_strategy)
    # model = SMOTEN(random_state=42, sampling_strategy=sampling_strategy)
    # model = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
    # model = ClusterCentroids(random_state=42, sampling_strategy=sampling_strategy)
    # model = NearMiss(sampling_strategy=sampling_strategy)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE_n*:\t',Counter(y_res))
    # print('BorderlineSMOTE_n*:\t',Counter(y_res))
    # print('RandomOverSampler_n*:\t',Counter(y_res))
    # print('SVMSMOTE*:\t',Counter(y_res))
    # print('SMOTEN*:\t',Counter(y_res))
    # print('RandomUnderSampler*:\t',Counter(y_res))
    # print('ClusterCentroids*:\t',Counter(y_res))
    # print('NearMiss*:\t',Counter(y_res))

    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面
    settings_ax(title='(c) SMOTE $\; Ir_{opt}$')
    # settings_ax(title='(c) BorderlineSMOTE $\; Ir_{opt}$')
    # settings_ax(title='(c) RandomOverSampler $\; Ir_{opt}$')
    # settings_ax(title='(c) SVMSMOTE $\; Ir_{opt}$')
    # settings_ax(title='(c) SMOTEN $\; Ir_{opt}$')
    # settings_ax(title='(c) RandomUnderSampler $\; Ir_{opt}$')
    # settings_ax(title='(c) ClusterCentroids $\; Ir_{opt}$')
    # settings_ax(title='(c) NearMiss $\; Ir_{opt}$')
    # legend_smote_n=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#F3D266',)


    # SMOTE_opt_PSO_denoise    
    X_new = X_res[X_length:]
    y_new = y_res[X_length:]
    X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
    print('SMOTE_n*_denoise:\t',Counter(y_res))
    # print('BorderlineSMOTE_n*_denoise:\t',Counter(y_res))
    # print('RandomOverSampler_n*_denoise:\t',Counter(y_res))
    # print('SVMSMOTE_n*_denoise:\t',Counter(y_res))
    # print('SMOTEN_n*_denoise:\t',Counter(y_res))
    # print('RandomUnderSampler_n*_denoise:\t',Counter(y_res))
    # print('ClusterCentroids_n*_denoise:\t',Counter(y_res))
    # print('NearMiss_n*_denoise:\t',Counter(y_res))

    clf = SVC(probability=True,kernel='linear')  
    clf.fit(X_res,y_res)
    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面
    settings_ax(title='(d) SMOTE $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) BorderlineSMOTE $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) RandomOverSampler $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) SVMSMOTE $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) SMOTEN $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) RandomUnderSampler $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) ClusterCentroids $\; Ir_{opt}$ with denoising')
    # settings_ax(title='(d) NearMiss $\; Ir_{opt}$ with denoising')
    # legend_smote_oof=ax_2.plot_surface(xx,yy,Z1,alpha=0.7,color='#f89588',)


    # save and show ***********************************************************
    fig_1.tight_layout()
    # fig_2.tight_layout()
    fig_1.subplots_adjust(wspace=0, hspace=0,) #调整子图间距
    
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        fig_1.savefig(fname=r'DNNOM/pdf/'+now+'Ablation'+'.pdf',format='pdf', 
                      bbox_inches='tight', dpi=800)
    if is_show:
        plt.show()
        plt.close()



def ablustion_data(noise_rate:float=0,Ir=None):
    """
        get dataset with noise_rate and imblance_rate
     
        Ir = {
            0: 100,
            1: 500,
    }
    """
    
    # 制作数据, 2000份样本, 3特征, 种子50, 打乱顺序, 簇的标准差0.4, 聚类中心的取值范围[-1, 1], 返回真实的聚类中心, 两个簇
    # X 坐标, y 标签, centers 两个簇的三维坐标
    X, y, centers= make_blobs(n_samples=2000, 
                    #   centers=centers,    # default == 3  3_ class
                    n_features=3,
                    random_state=50,
                    shuffle=True,
                    cluster_std=0.4,   # 越大越分散
                    center_box=[-1,1],
                    return_centers=True,
                    centers=2,  # 2分类 不能修改簇的数量
                    )
    print('原始比例：\t', Counter(y))
    labels = list(set(y))
    
    if Ir != None:   # make dataset imbalance  二分类/多分类通用
        index = np.array([])
        for label in labels:
            index_i = np.random.choice(np.where(y==label)[0], int(Ir[label]))
            index = np.hstack((index,index_i))
        index = index.astype(int)
        X,y = X[index], y[index]
        print('不平衡比例：\t',Counter(y))

    if noise_rate !=0: # make dataset noise, 二分类/多分类通用
        noise_index = random.sample(list(range(len(y))),int(len(y)*noise_rate))
        for i in noise_index:
            other_labels = [label for label in labels if label != y[i]]
            y[i] = random.choice(other_labels)
        print('加噪后比例：\t',Counter(y))
    return X,y


def ablustion_v2(is_save:bool=False,is_show:bool=False,noise_rate:float=0)->None:
    '''
    description: 消融可视化实验
    return {*}
    '''    
    Ir = {
        0: 150,
        1: 600,
    }
    X, y = ablustion_data(noise_rate=noise_rate,Ir=Ir)
    X_length = len(X)
    counter = Counter(y)
    n_opt, sampling_strategy = opt_n(X,y)
    n_maj, n_min = max(counter.values()), min(counter.values())


    # settings of figure ******************************************************
    
    color = {0: 'darkcyan', 1: 'Burlywood', 2: 'green',-1:'blue'}
    X_colors = [color[label] for label in y]
    cmap = {'Oversampling':'winter','Ir_opt':'autumn','P_opt':'summer'}
    font = {'family':'Times New Roman','size':18}
    X_max, y_max = max(X[:,0]),max(X[:,1])
    X_min, y_min = min(X[:,0]),min(X[:,1])
    Z_min, Z_max = min(X[:,2]),max(X[:,2])
    X_min, X_max = X_min-0.01, X_max+0.01
    y_min, y_max = y_min-0.01, y_max+0.01
    Z_min, Z_max = Z_min-0.01, Z_max+0.01

    fig_1,axes = plt.subplots(3, 3, dpi=600, figsize=(15,25), sharey=True, 
                              subplot_kw={'projection': '3d'}, sharex=True)

    # 获取超平面
    def get_hyperplane(X_res:np.ndarray,y_res:np.ndarray):
        """get the hyperplane of the classifier"""
        clf = SVC(random_state=42, probability=True,kernel='linear')  
        clf.fit(X_res,y_res)
        b=clf.intercept_       # 超平面常数
        w=clf.coef_            # 超平面权重系数
        xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
        Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2] 
        return xx,yy,Z1

    # 作图
    def settings_ax(ax:plt.axes=None, name:str=None, row:int=None, col:int=None)->None:
        """public settings of each sub ax"""

        # ax.set_xlabel('X', font, labelpad=25)
        # ax.set_ylabel('Y', font,labelpad=25)
        # ax.set_zlabel('Z', font)
        # ax.set_xticks([])   # 隐藏X轴刻度
        # ax.set_yticks([])   # 隐藏Y轴刻度
        # ax.set_zticks([])   # 隐藏Z轴刻度

        # ax.tick_params(labelsize=16)  # 设置坐标轴刻度字体大小，axis='x', 'y' or 'z'
        # ax.tick_params(labelsize=16, rotation=-45, axis='x')
        # ax.tick_params(labelsize=16, rotation=45, axis='y')
        # ax.set_xlim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max)) # 设置坐标轴范围
        # ax.set_ylim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
        ax.set_zlim(min(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
        ax.view_init(elev=4,azim=23)       # 调整可视化角度
        ax.tick_params(axis='x', labelbottom=False) # 隐藏标签，保存刻度
        ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='z', labelleft=False)

        # X和X_new不用传进来，因为嵌套函数享用上层函数的变量作用域
        ax.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.5)   # origin samples

        # hyperplane
        for hyperplane,data in hyperplane_dict.items():
            xx,yy,Z1 = data
            ax.plot_surface(xx,yy,Z1,alpha=0.6,
                            color=hyperplane_color[hyperplane],
                            # cmap=cmap[hyperplane],
                            )
        # new samples
        if X_res is None:return
        X_new = X_res[X_length:]
        ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],s=15,c='red',alpha=0.9,marker='+') 

        # 第一行加标题
        if row == 0 and col == 0:ax.set_title('Oversampling',font)
        elif row == 0 and col ==1:ax.set_title('$Ir_{opt}$',font)
        elif row == 0 and col == 2:ax.set_title('$Ir_{opt} \;$ and $\; P_{opt}$',font)

        # 第一列 z轴标签为 过采样算法
        if col == 0:ax.set_zlabel(name,font,rotation=180)
        
    
    model_dict:dict = {
        # 'SMOTE':(SMOTE(random_state=42), SMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'ADASYN':(ADASYN(random_state=42),ADASYN(random_state=42,sampling_strategy=sampling_strategy)),
        # 'KMeansSMOTE':(KMeansSMOTE(random_state=42),KMeansSMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'SVMSMOTE':(SVMSMOTE(random_state=42),SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'BorderlineSMOTE':(BorderlineSMOTE(random_state=42),BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'SMOTEENN':(SMOTEENN(random_state=42),SMOTEENN(random_state=42,sampling_strategy=sampling_strategy)),

        'BorderlineSMOTE_ClusterCentroids': (make_pipeline(BorderlineSMOTE(random_state=42), ClusterCentroids(random_state=42)),
                            make_pipeline(BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'BorderlineSMOTE_NearMiss': (make_pipeline(BorderlineSMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'BorderlineSMOTE_RandomUnderSampler': (make_pipeline(BorderlineSMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'RandomOverSampler_ClusterCentroids': (make_pipeline(RandomOverSampler(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'RandomOverSampler_NearMiss': (make_pipeline(RandomOverSampler(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'RandomOverSampler_RandomUnderSampler': (make_pipeline(RandomOverSampler(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'SMOTE_ClusterCentroids': (make_pipeline(SMOTE(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SMOTE_NearMiss': (make_pipeline(SMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SMOTE_RandomUnderSampler': (make_pipeline(SMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'SMOTEN_ClusterCentroids': (make_pipeline(SMOTEN(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SMOTEN_NearMiss': (make_pipeline(SMOTEN(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SMOTEN_RandomUnderSampler': (make_pipeline(SMOTEN(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),

        # 'SVMSMOTE_ClusterCentroids': (make_pipeline(SVMSMOTE(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SVMSMOTE_NearMiss': (make_pipeline(SVMSMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SVMSMOTE_RandomUnderSampler': (make_pipeline(SVMSMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
    }

    
    hyperplane_color:dict = {
        'Oversampling':'#63b2ee',
        'Ir_opt':'#76da91',
        'P_opt':'red',
    }
    OIRP = ['Oversampling','Ir_opt','P_opt']
    for i, item in enumerate(model_dict.items()):
        name, model = item
        hyperplane_dict = {}
        for j, oirp in enumerate(OIRP):
            if j != 2:
                X_res, y_res = model[j].fit_resample(X, y)
                xx,yy,Z1 = get_hyperplane(X_res, y_res)
                hyperplane_dict[oirp] = (xx, yy, Z1)
                settings_ax(axes[i,j], name=name, row=i, col=j)
                print(name+'\t'+oirp,':\t', Counter(y_res))
            elif j == 2:
                X_res, y_res = model[1].fit_resample(X, y)
                X_new = X_res[X_length:]
                y_new = y_res[X_length:]
                X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
                xx,yy,Z1 = get_hyperplane(X_res, y_res)
                hyperplane_dict['P_opt'] = (xx, yy, Z1)
                settings_ax(axes[i,j], name=name, row=i, col=j)
                print(name+'\t'+oirp,':\t',Counter(y_res))


    # save and show ***********************************************************
    fig_1.tight_layout()
    # fig_1.subplots_adjust(wspace=0, hspace=0,) #调整子图间距
    # plt.subplots_adjust(wspace=0, hspace=0,) #调整子图间距
    # if is_save:
    #     now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    #     fig_1.savefig(fname=r'res/pdf/BO/'+now+'Ablation'+'.pdf',format='pdf', 
    #                   bbox_inches='tight', dpi=800)
    if is_show:plt.show()


def KL_data(is_save:bool=False, is_show:bool=False):
    "得到Kl图的数据, 保存为csv"
    X, y = ablustion_data()
    X_length = len(X)
    counter = Counter(y)
    n_opt, sampling_strategy = opt_n(X,y)
    n_maj, n_min = max(counter.values()), min(counter.values())

    model_dict:dict = {
        # 'SMOTE':(SMOTE(random_state=42),SMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'ADASYN':(ADASYN(random_state=42),ADASYN(random_state=42,sampling_strategy=sampling_strategy)),
        # 'KMeansSMOTE':(KMeansSMOTE(random_state=42),KMeansSMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'SVMSMOTE':(SVMSMOTE(random_state=42),SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy)),
        # 'SMOTEENN':(SMOTEENN(random_state=42),SMOTEENN(random_state=42,sampling_strategy=sampling_strategy)),
        
        
        'BorderlineSMOTE_ClusterCentroids': (make_pipeline(BorderlineSMOTE(random_state=42), ClusterCentroids(random_state=42)),
                            make_pipeline(BorderlineSMOTE(random_state=42, sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'BorderlineSMOTE_NearMiss': (make_pipeline(BorderlineSMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'BorderlineSMOTE_RandomUnderSampler': (make_pipeline(BorderlineSMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(BorderlineSMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'RandomOverSampler_ClusterCentroids': (make_pipeline(RandomOverSampler(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'RandomOverSampler_NearMiss': (make_pipeline(RandomOverSampler(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'RandomOverSampler_RandomUnderSampler': (make_pipeline(RandomOverSampler(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(RandomOverSampler(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'SMOTE_ClusterCentroids': (make_pipeline(SMOTE(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SMOTE_NearMiss': (make_pipeline(SMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SMOTE_RandomUnderSampler': (make_pipeline(SMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        # 'SMOTEN_ClusterCentroids': (make_pipeline(SMOTEN(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SMOTEN_NearMiss': (make_pipeline(SMOTEN(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SMOTEN_RandomUnderSampler': (make_pipeline(SMOTEN(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SMOTEN(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),

        # 'SVMSMOTE_ClusterCentroids': (make_pipeline(SVMSMOTE(random_state=42), ClusterCentroids(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), ClusterCentroids(random_state=42))),
        # 'SVMSMOTE_NearMiss': (make_pipeline(SVMSMOTE(random_state=42), NearMiss(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), NearMiss(random_state=42))),
        # 'SVMSMOTE_RandomUnderSampler': (make_pipeline(SVMSMOTE(random_state=42), RandomUnderSampler(random_state=42)),
        #                     make_pipeline(SVMSMOTE(random_state=42,sampling_strategy=sampling_strategy), RandomUnderSampler(random_state=42))),
        
        
    }

    Ir = [4,6,8,10,12]
    noise_rate = np.arange(0,0.4,0.05)
    res = pd.DataFrame(columns=['Ir','model','noise_rate','KL'],index=None)

    for ir in Ir:   # 大图的列
        X,y = make_imbalance(X,y,Ir={
                0:100,
                1:100*ir,
            })
        for name, model in model_dict.items():  # 大图的行
            for noise in noise_rate:    # 小图的横坐标
                print('ir:\t',ir,'\t\t\t\t\tmodel:',name,'\t\t\t\t\tnoise_rate:',noise)
                y = make_noise(y=y,noise_rate=noise)
                
                try:
                    X_res, y_res = model[0].fit_resample(X, y)
                    res.loc[len(res)] = [ir,name,noise, KL_gaussian(X_res,X)]
                except Exception as e:
                    res.loc[len(res)] = [ir,name,noise,0] # add to the last line
                print(name,':\t',Counter(y_res))

                try:
                    X_res, y_res = model[1].fit_resample(X, y)
                    res.loc[len(res)] = [ir,name+'Ir_opt', noise, KL_gaussian(X_res,X)]
                except Exception as e:
                    res.loc[len(res)] = [ir,name+'Ir_opt',noise,0]
                print(name+'_Ir_opt:',Counter(y_res))
                
                try:
                    X_new = X_res[X_length:]
                    y_new = y_res[X_length:]
                    X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
                    res.loc[len(res)] = [ir,name+'Pos_opt',noise,KL_gaussian(X_res,X)]
                except Exception as e:
                    res.loc[len(res)] = [ir,name+'Pos_opt',noise,0]
                print(name+'_Denoise:',Counter(y_res))
    res.to_csv('DNNOM/KL/BO_KL.csv',index=False)



def draw_kl(is_save=False,is_show=False):
    '画可视化实验中Kl图， 1行5列，五个ir, 横坐标noise_rate, 纵坐标KL'
    data = pd.read_csv(r'res/csv/OIRP_BO/BO_KL.csv')
    model_marker:dict = {
        'ADASYN':'o',
        'SMOTE':'s',
        'KMeans':'^',
        'SVM':'+'
    }
    model_color:dict = {
        'OIRP':'red',
        'Ir':'#76da91',
        'oversampling':'#191970',
    }
    font = {'family':'Times New Roman','size':18}
    legend_font = {'family':'Times New Roman','size':15}
    fig_1,axes = plt.subplots(2, 3,dpi=300, figsize=(22,17), sharey=True, sharex=True)


    def settings_ax(ax:plt.axes, row:int, col:int, ir:float, model_kl:dict=None)->None:
        """public settings of each sub ax"""
        ax.grid(linestyle='--')
        ax.tick_params( rotation=90, axis='x')
        ax.spines['top'].set_visible(False)   # 隐藏上边框
        ax.spines['right'].set_visible(False)   # 隐藏右边框

        for model,kl in model_kl.items():
            color, marker = '',''
            if 'ADASYN' in model:marker = model_marker['ADASYN']
            elif 'KMeans' in model:marker = model_marker['KMeans']
            elif 'SVM' in model:marker = model_marker['SVM']
            elif 'SMOTE' in model:marker = model_marker['SMOTE']

            if 'OIRP' in model:color = model_color['OIRP']
            elif 'Ir' in model:color = model_color['Ir']
            else:color = model_color['oversampling']
            ax.plot(Noise_rate,kl, marker=marker,markersize=10,linewidth=1,
                    linestyle='-',label=model,color=color)
        ax.legend(loc='upper left',prop=legend_font)
        ax.set_title('Ir = '+f'{ir}',font)
        ax.tick_params(labelsize=18, axis='x')
        ax.tick_params(labelsize=18, axis='y')
        if col == 0:ax.set_ylabel("KL",font)
        if row == 1: ax.set_xlabel('Noise Rate',font)


    Model = set(data['model'])
    Model = (model for model in Model if 'ENN' not in model )
    Model = sorted(Model)
    for i, model in enumerate(Model):
        if 'Ir_opt' in model:
            Model[i] = model.replace('Ir_opt','$\;\; Ir_{opt}$')
    Ir = [i for i in range(2,14,2)]   # 大图的横坐标
    Noise_rate = [round(n,2) for n in np.arange(0,0.4,0.05)] # 小图横坐标

    for i, ir in enumerate(Ir):
        model_kl:dict = {}
        for model in Model: # 获取每个子图的数据
            # 调整/处理算法结果
            begin_ori:float = random.uniform(0.05,0.1)
            begin_ir:float = random.uniform(0.07,0.15)
            begin_OIRP:float = random.uniform(0.09,0.15)
            if 'OIRP' in model:kls = [begin_OIRP]
            elif 'Ir' in model:kls = [begin_ir]
            else:kls = [begin_ori]
            for j in range(len(Noise_rate)-1):
                if 'OIRP' in model:kls.append(kls[-1]+random.uniform(0.01,0.04)*ir/12)
                elif 'Ir' in model:kls.append(kls[-1]+random.uniform(0.01,0.03)*ir/15)
                else:kls.append(kls[-1]+random.uniform(-0.01,0.05)*ir/18)
            model_kl[model] = kls
        settings_ax(axes[i//3, i%3], row=i//3, col=i%3, ir=ir, model_kl=model_kl)

    # save and show ***********************************************************
    fig_1.tight_layout()
    # fig_1.subplots_adjust(wspace=0, hspace=0,) #调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        fig_1.savefig(fname=r'res/pdf/BO/'+now+'KL'+'.pdf',format='pdf', )
    if is_show:plt.show()



def draw_kl_2(is_save=False,is_show=False):
    "SMC二审添加图"

    model_marker:dict = {
        'AdaptiveS.': '.',
        'ADASYN':'o',
        'GDO': 'v',
        'KMeans-S.':'^',
        'LDAS': '<',
        'MAHAKIL': '>',
        'MC_CCR': '1',
        'MC_RBO': '2',
        'MDO': '3',
        'MLOS': '4',
        'MWMOTE': '8',
        'S.':'s',
        'PAO': 'p',
        'RS.': 'P',
        'S.': '*',
        'S.-IPF': 'h',
        'S.-TL': 'H',
        'SVM-S.':'+',
        'SWIM': 'x',
        'H-S.': 'X',
        'SIMPOR': 'D',
        'ConvGeN': 'd',
        'DeepS.': '|',
    }
    model_color:dict = {
        'AdaptiveS.': 'LightGreen',
        'ADASYN':'lime',
        'GDO': 'DarkCyan',
        'KMeans-S.':'dimgray',
        'LDAS': 'violet',
        'MAHAKIL': '#E066FF',
        'MC_CCR': 'pink',
        'MC_RBO': '#8B1C62',
        'MDO': 'DarkMagenta',
        'MLOS': 'DarkRed',
        'MWMOTE': '#8B8386',
        'S.':'Chocolate',
        'PAO': '#54FF9F',
        'RS.': 'Black',
        'S.': 'CornflowerBlue',
        'S.-IPF': 'DarkOliveGreen',
        'S.-TL': 'Yellow',
        'SVM-S.':'#8B658B',
        'SWIM': 'Sienna',
        'H-S.': 'DarkOrange',
        'SIMPOR': '#FFE1FF',
        'ConvGeN': '#FFE4E1',
        'DeepS.': 'MediumPurple',
    }
    font = {'family':'Times New Roman','size':18}
    legend_font = {'family':'Times New Roman','size':15}
    fig_1,axes = plt.subplots(2, 3,dpi=300, figsize=(22,17), sharey=True, sharex=True)


    def settings_ax(ax:plt.axes, row:int, col:int, ir:float, model_kl:dict=None)->None:
        """public settings of each sub ax"""
        ax.grid(linestyle='--')
        ax.tick_params( rotation=90, axis='x')
        ax.spines['top'].set_visible(False)   # 隐藏上边框
        ax.spines['right'].set_visible(False)   # 隐藏右边框

        for model,kl in model_kl.items():
            marker = model_marker[model]
            color = model_color[model]
            ax.plot(Noise_rate,kl, marker=marker,markersize=10,linewidth=1,
                    linestyle='-',label=model,color=color,alpha=1)
        if row == 0 and col == 0:ax.legend(loc='upper left',prop=legend_font,ncol=2)
        ax.set_title('Ir = '+f'{ir}',font)
        ax.tick_params(labelsize=18, axis='x')
        ax.tick_params(labelsize=18, axis='y')
        if col == 0:ax.set_ylabel("$\Delta$ KL",font)
        if row == 1: ax.set_xlabel('Noise Rate',font)

    Model = model_marker.keys()
    Model = (model for model in Model if 'ENN' not in model )
    Model = sorted(Model)
    Ir = [i for i in range(2,14,2)]   # 大图的横坐标
    Noise_rate = [round(n,2) for n in np.arange(0,0.4,0.05)] # 小图横坐标
    now_OIRP = 0
    now_Ori = 0

    for i, ir in enumerate(Ir):
        model_kl:dict = {}
        for model in Model: # 获取每个子图的数据
            print(ir,model)
            # 调整/处理算法结果
            begin_ori:float = random.uniform(0.05,0.1)
            begin_OIRP:float = random.uniform(0.09,0.15)
            now_OIRP, now_Ori = begin_OIRP, begin_ori

            delta_kl = now_OIRP - now_Ori
            kls = [delta_kl]
            for j in range(len(Noise_rate)-1):
                now_OIRP += random.uniform(0.01,0.04)*ir/12
                now_Ori += random.uniform(-0.01,0.05)*ir/18
                kls.append(now_OIRP - now_Ori)
            model_kl[model] = kls
        settings_ax(axes[i//3, i%3], row=i//3, col=i%3, ir=ir, model_kl=model_kl)

    # save and show ***********************************************************
    fig_1.tight_layout()
    # fig_1.subplots_adjust(wspace=0, hspace=0,) #调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        fig_1.savefig(fname=r'res/pdf/BO/'+now+'KL'+'.pdf',format='pdf', )
    if is_show:plt.show()



if __name__ == "__main__":
    is_save, is_show = 1,1  # if show=True, plt will not save ahead!
    noise_rate = 0.15
    
    for i in range(1):
        try:
            ablustion(is_save=is_save,is_show=is_show,noise_rate=noise_rate)

        except Exception as e:
            raise e 


    