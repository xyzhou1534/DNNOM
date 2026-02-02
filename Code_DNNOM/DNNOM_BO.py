# import system package
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from collections import Counter
import random
from scipy.stats import multivariate_normal
# from sympy import *
# from latex2sympy2 import latex2sympy, latex2latex
from scipy.optimize import minimize
from api import binary_data
import random
import math
from imblearn.over_sampling import SMOTE


# import Diy packages
# from compare_methods.Classification.Over.INGB import INGB
from api import binary_data


def PSO_Denoise(X_origin:np.ndarray, y_origin, X_min_gen, y_min_gen)->np.ndarray:
    """
        将所有新样本当作一个粒子
        初始粒子群是初始粒子的复制体
        迭代num_it,取全局最优
    """

    def loss_function(X_train, y_train, X_test, y_test):
        svc= SVC(kernel= 'rbf', probability= True, gamma= 'auto', 
                 random_state=settings['random_state'])
        svc.fit(X_train, y_train)
        y_pred= svc.predict_proba(X_test)[:,np.where(svc.classes_ == minority_label)[0][0]]
        return roc_auc_score(y_test, y_pred, multi_class='ovr')

    # base settings
    settings = {
        'eps':0.05,
        'n_pop': 20,     # population size
        'w': 0.3,
        'c1': 0.5,
        'c2': 0.5,
        'num_it': 10,     # iteration times
        'random_state':42,
        'random_float':random.random(),
    }

    nums,dims = X_min_gen.shape
    count = Counter(y_origin)
    minority_label = min(count, key=count.get)  
    # majority_label = max(count, key=count.get)

    # get the insearch space
    min_vector = []
    maj_vector = []
    search_space= []        
    init_velocity= []
    X_origin_minority = X_origin[np.where(y_origin==minority_label)[0]]    

    for d in range(dims):  # 每个维度的最值
        #TODO:边界改成新样本的最值
        up_bound, down_bound = max(X_origin_minority[:, d]), min(X_origin_minority[:, d])
        min_vector.append(down_bound)
        maj_vector.append(up_bound)
    
    min_vector, maj_vector = np.array(min_vector), np.array(maj_vector)
    init_velocity.append(settings['eps']*(maj_vector-min_vector))
    search_space.append([min_vector, maj_vector, np.linalg.norm(min_vector - maj_vector)])
    init_velocity = np.array(init_velocity,dtype=object) #  change type to np
    search_space = np.array(search_space,dtype=object)
    
    init_velocity = np.repeat(init_velocity,nums,axis=0)
    search_space = np.repeat(search_space,nums,axis=0)

    # initializing the particle swarm and the particle and population level
    particle_swarm = [X_min_gen.copy() for _ in range(settings['n_pop'])]
    velocities= [init_velocity.copy() for _ in range(settings['n_pop'])]
    local_best= [X_min_gen.copy() for _ in range(settings['n_pop'])]
    local_best_scores= [0.0]*settings['n_pop']
    global_best= X_min_gen.copy()
    global_best_score= 0.0 

    # begin to generate
    for i in range(settings['num_it']):
        # evaluate population
        scores= [loss_function(np.vstack([X_origin, p]), 
                np.hstack([y_origin, np.repeat(minority_label, len(p))]), 
                X_origin, y_origin) for p in particle_swarm]
        
        # update best scores
        for i, s in enumerate(scores):
            if s > local_best_scores[i]:
                local_best_scores[i]= s
                local_best[i]= particle_swarm[i]
            if s > global_best_score:
                global_best_score= s
                global_best= particle_swarm[i]
        
        # update velocities
        for i, p in enumerate(particle_swarm):
            """
            v_{ij}(t + 1) = 
                    w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                    + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]
            """
            velocities[i]= settings['w']*velocities[i] 
            + settings['c1']*random.random()*(local_best[i] - p) 
            + settings['c2']*random.random()*(global_best - p)
        
        # bound velocities according to search space constraints
        for v in velocities:
            for i in range(len(v)):
                if np.linalg.norm(v[i]) > search_space[i][2]/2.0:
                    v[i]= v[i]/np.linalg.norm(v[i])*search_space[i][2]/2.0
        
        # update positions
        for i, p in enumerate(particle_swarm):
            """x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)"""
            particle_swarm[i]= particle_swarm[i] + velocities[i]
        
        # bound positions according to search space constraints
        for p in particle_swarm:
            for i in range(len(p)):
                if np.linalg.norm(p[i] - search_space[i][0]) > search_space[i][2]:
                    p[i]= search_space[i][0] + (p[i] - search_space[i][0])/np.linalg.norm(p[i] - search_space[i][0])*search_space[i][2]

    return np.vstack([X_origin, global_best]),     \
            np.hstack([y_origin, np.repeat(minority_label, len(global_best))])



def opt_n(X:np.ndarray,y:np.ndarray):
    '''description: 传入数据集，返回最佳采样数量和比例'''

    # 检查数据格式
    assert type(X)==np.ndarray
    assert type(y)==np.ndarray
    assert len(X)==len(y)


    def gaussian_density_based_obj_fun(n:int)->float:
        """基于多元高斯分布的目标函数"""
        
        # 计算每个样本向量的多元高斯分布密度
        density = 0 # initialize the density
        for x in X:
            diff = x - mean # 计算与均值向量的差
            inv_cov = np.linalg.inv(cov) # 计算协方差矩阵的逆矩阵

            # objective-function 凹转凸  变量n
            density += 1 / (np.sqrt(det_cov * (2 * np.pi) ** dims)) * \
                  np.exp(-0.5 * np.dot(np.dot(diff.T, inv_cov), diff))  \
                + random.uniform(0, 3) *np.linalg.norm(n*x)
        return density # sum of Gaussian density of the dataset
    

    # infomation of the dataset =======================================
    column_means = np.mean(X, axis=0)   # 每列均值
    mean_vector = column_means.reshape(1, -1)   # 均值向量
    mean = mean_vector[0]   # 均值向量
    centered_data = X - mean_vector # 计算每一行与均值的差
    covariance_matrix = np.cov(centered_data.T)  # 协方差矩阵
    cov = covariance_matrix
    det_cov = np.linalg.det(cov)    # 计算协方差矩阵的行列式

    density = multivariate_normal.pdf(X, mean=mean, cov=covariance_matrix)
    densities = sum(density)    # 数据集原始高斯密度总和

    count = Counter(y)
    num_maj, num_min = max(count.values()), min(count.values())
    label_maj, label_min = max(count,key=count.get), min(count,key=count.get)
    Ir = num_maj/num_min    # Imbalanced rate
    total_num,dims = X.shape 

    # begin to optmize n =================================================
    e = 1e-10
    cons = (
        {'type': 'ineq','fun': lambda n: n-e},
        {'type': 'ineq','fun': lambda n: (num_maj-num_min)-n-e},
        )
    
    n0 = abs(int(math.sin(densities + dims*num_min/num_maj) * (num_maj-num_min)))  
    
    res = minimize(gaussian_density_based_obj_fun,n0,
                   method='SLSQP',
                   constraints=cons)
    # print("最小值:",res.fun)
    # print("最优解:",res.x,int(res.x))
    # print("迭代终止是否成功",res.success)
    # print("迭代终止原因",res.message)
    return (
            int(res.x),
            # int(res.x)+num_min, 
            float((res.x+num_min)/num_maj)
            ) 



if __name__ == '__main__':
    X, y = binary_data(data_name='make_moons')
    counter = Counter(y)
    print('original rate', Counter(y))
    
    # compute n* ============================
    n_opt,sampling_strategy = opt_n(X,y) 
    print("最优解n*, :\t sampling_strategy:\t",n_opt,sampling_strategy)

    model = SMOTE(random_state=42,sampling_strategy=sampling_strategy)
    X_res, y_res = model.fit_resample(X, y)
    print('SMOTE n* Oversampling:\t',Counter(y_res))

    X_length = len(X)
    X_new = X_res[X_length:]
    y_new = y_res[X_length:]
    X_res,y_res = PSO_Denoise(X.copy(),y.copy(),X_new.copy(),y_new.copy())
    print('SMOTE_opt_PSO_denoise:\t',Counter(y_res))