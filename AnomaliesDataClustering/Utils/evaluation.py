from sklearn import metrics
from sklearn.metrics import f1_score


# 有监督评价指标：F1值
def f1_score(labels_true, labels_pred):
    return f1_score(labels_true, labels_pred, average="micro")


# 有监督评价指标：调整后的兰德系数
# 取值在［－1，1］之间，负数代表结果不好，越接近于1越好
def ari_score(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)


# 有监督评价指标：调整后的互信息
# 取值范围在［0，1］之间,越大越好
def ami_score(labels_true, labels_pred):
    return metrics.adjusted_mutual_info_score(labels_true, labels_pred)


# 无监督评价指标：轮廓系数
# 取值范围在［-1，1］之间,越大越好
def sc_score(datas, labs):
    return metrics.silhouette_score(datas, labs, metric='euclidean',
                                    sample_size=None, random_state=None)


# 无监督评价指标：CH分数
# CH分数越大越好
def ch_score(datas, labs):
    return metrics.calinski_harabasz_score(datas, labs)


# 无监督评价指标：DBI
# DBI的值最小是0，值越小，代表聚类效果越好
def dbi_score(datas, labs):
    return metrics.davies_bouldin_score(datas, labs)

