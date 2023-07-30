# 本文件存储聚类算法所可能用到的各种距离度量方式
import numpy as np


# 计算欧氏距离（对应值的差平方之和再开方），注重数据之间的绝对位置而不是方向
# np.linalg.norm用于范数计算，默认二范数，相当于平方和开根号
def calsim_eud(data1, data2):
    return np.sqrt(np.dot((data1 - data2), (data1 - data2)))  # 未归一化处理
    # return 1 / (1 + np.linalg.norm(data1 - data2))  # 原始取值为(0,正无穷),归一化到(0,1]区间：1/(1+原始值）


# 计算余弦相似度，注重数据的方向而非绝对位置
# 已进行归一化，取值[0,1]，数值越大表示相似性越高，数值为1代表完全相似
def calsim_cosine(data1, data2):
    sumData = np.dot(data1, data2)  # 公式中的分子，向量的内积
    # np.linalg.norm用于范数计算，默认二范数，相当于平方和开根号
    denom = np.linalg.norm(data1) * np.linalg.norm(data2)  # 公式中的分母
    # 原始取值为[-1,1]，归一化到[0,1]区间：0.5 + 0.5 * 原始值
    return 0.5 + 0.5 * (sumData / denom)


# 计算皮尔逊相关系数，是对余弦相似度的修正，分子和分母都需要减去输入数据集各自本身向量的均值，以达到中心化
# 已进行归一化，取值[0,1]，数值越大表示相似性越高，数值为1代表完全相似
def calsim_pearson(data1, data2):
    # np直接计算出的皮尔逊相关系数取值范围[-1,1]，归一化到[0,1]区间：0.5 + 0.5 * 原始值
    return 0.5 + 0.5 * np.corrcoef(data1, data2, rowvar=0)[0][1]


# 计算Jaccard相似度，度量集合之间的差异，共有的元素越多则越相似
# 取值[0,1]，数值越大表示相似性越高，数值为1代表完全相似
def calsim_jaccard(data1, data2):
    a_len, b_len = len(data1), len(data2)
    c = [i for i in data1 if i in data2]  # 取交集
    c_len = len(c)  # 交集含有元素的个数
    return c_len / (a_len + b_len - c_len)
