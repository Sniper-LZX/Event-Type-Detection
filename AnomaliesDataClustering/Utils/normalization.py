import numpy as np


# Min-Max归一化：
# 使用场景：不涉及距离度量、协方差计算时，数据不符合正态分布时使用。
def min_max_normaliztion(datas):
    x_min, x_max = np.min(datas, 0), np.max(datas, 0)
    new_datas = datas / (x_max - x_min)
    return new_datas


# Z-Score归一化：处理后的数据将服从标准正态分布，即均值为0，标准差为1
# 使用场景：原始数据近似为正态分布时使用，在分类、聚类算法中，或使用PCA进行降维时使用。
def z_score_normalization(datas):
    zscored_datas = (datas - np.mean(datas)) / np.std(datas)
    return zscored_datas