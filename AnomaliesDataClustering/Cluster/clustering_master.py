from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from Cluster import dbscan_home_made as db_hm
from Cluster import SOM_home_made as som
from Utils import parameter_optimization as po


# DBSCAN算法实现，分为sklearn实现与自定义实现两种方式
# datas: 将要聚类的数据
# eps: 邻域半径，邻域半径越大，所聚类别就会越少，聚类更集中
# min_points: 核心点邻域内最少点个数，取值越大，所聚类别越少
def dbscan(datas):
    print("当前聚类算法选择DBSCAN")
    # 绘制k-distance图，确定DBSCAN算法最佳eps邻域半径取值
    # k值一般取 2 * 维度 - 1
    # value: 取正整数，用来辅助在k-dist图上标明拐点位置，eps取拐点位置值最为合理
    # 如果value偏离拐点，增大value标记点向右移，减小向左移
    # mec:是否开启小数点后细微误差控制，mec=0默认不开启, >0则开启
    k = 2 * datas.ndim - 1
    eps_recommend = po.k_dist_mapping(k, datas, value=20, mec=1)

    # eps:邻域半径； min_points:核心点邻域内最少点个数
    # eps_recommend值为推荐值，程序可依据误差，对推荐值进行人工微调
    eps = eps_recommend
    min_points = 5

    # 使用sklearn包附带的DBSCAN算法
    skl_db = DBSCAN(eps=eps, min_samples=min_points).fit(datas)
    labs = skl_db.labels_
    raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目
    print("labs of sk-DBSCAN")

    # 使用自定义DBSCAN算法
    # labs, n_clusters = db_hm.dbscan(datas, eps=eps, min_points=min_points)
    # raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    # print("labs of my dbscan")

    return labs, n_clusters, raito


# k-means算法实现
# 使用轮廓系数，选取最佳的k值
def k_means(datas):
    print("当前聚类算法选择K-means")
    k = 19
    km = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(datas)
    labs = km.labels_
    raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目

    # best_k = []
    # N = 10 # 最多分十类
    # for n in range(2, N + 1):  # 从2-10中寻找最佳k值
    #     km = KMeans(n_clusters=n, init='k-means++', random_state=0).fit(datas)
    #     labs = km.labels_
    #     evaluate_res = metrics.silhouette_score(datas, labs)
    #     best_k.append((evaluate_res, n, labs))
    #
    # best_k = sorted(best_k, key=lambda x: x[0])[-1]
    # evaluate_res, n_clusters, labs = best_k
    # raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    return labs, n_clusters, raito


# 层次聚类算法实现
# 使用轮廓系数选取最合适的聚簇数量
def hierarchical_clustering(datas):
    print("当前聚类算法选择层次聚类")
    """
        参数:
            n_cluster:聚类数量    affinity:距离度量方法，可选 ‘euclidean’, ‘manhattan’,‘l1’,‘l2’,‘cosine’,‘precomputed’
            linkage:选择何种距离，可选’ward'（组间距离等于两类对象之间的最小距离），‘complete'（组间距离等于两组对象之间的最大距离）,'average'（组间距离等于两组对象之间的平均距离）,'single'(最近距离）
            distance_threshold:距离阈值，大于这个阈值后，不会合并
            compute_full_tree:是否生成一颗完整的树，设置成否可以节省计算开销
        属性：
            labels_ 聚类结果
    """
    model = AgglomerativeClustering(n_clusters=19, linkage='ward')
    model.fit(datas)  # 训练模型
    labs = model.labels_
    n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目
    raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例

    # best_n = []
    # N = 10  # 最多分十类
    # for n in range(2, N + 1):  # 从2-10中寻找最佳聚簇
    #     model = AgglomerativeClustering(n_clusters=n, linkage='ward').fit(datas)
    #     labs = model.labels_
    #     evaluate_res = metrics.silhouette_score(datas, labs)
    #     best_n.append((evaluate_res, n, labs))
    #
    # best_n = sorted(best_n, key=lambda x: x[0])[-1]
    # evaluate_res, n_clusters, labs = best_n
    # raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    return labs, n_clusters, raito


# 高斯混合模型聚类算法
# 使用轮廓系数选取最合适的聚簇数量
def gaussian_mixture(datas):
    print("当前聚类算法选择GMM高斯混合模型")
    model = GaussianMixture(n_components=5)
    labs = model.fit_predict(datas)
    n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目
    raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例

    # best_n = []
    # N = 10  # 最多分十类
    # for n in range(2, N + 1):  # 从2-10中寻找最佳聚簇
    #     model = GaussianMixture(n_components=n)
    #     labs = model.fit_predict(datas)
    #     evaluate_res = metrics.silhouette_score(datas, labs)
    #     print("当前: ", n, ",", evaluate_res)
    #     best_n.append((evaluate_res, n, labs))
    #
    # best_n = sorted(best_n, key=lambda x: x[0])[-1]
    # evaluate_res, n_clusters, labs = best_n
    # raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    return labs, n_clusters, raito


# 谱聚类
# 使用CH分数选取合适的聚簇数量
def spectral_clustering(datas):
    print("当前聚类算法选择谱聚类")
    # labs = SpectralClustering(n_clusters=5, gamma=0.1).fit_predict(datas)
    # raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    # n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目

    best_n = []
    N = 10  # 最多分十类
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        for n in range(2, N + 1):
            labs = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(datas)
            evaluate_res = metrics.calinski_harabasz_score(datas, labs)
            best_n.append((evaluate_res, n, labs, gamma))

    best_n = sorted(best_n, key=lambda x: x[0])[-1]
    print(best_n)
    evaluate_res, n_clusters, labs, gamma = best_n
    raito = len(labs[labs[:] == -1]) / len(labs)  # 计算噪声点个数占总数的比例
    return labs, n_clusters, raito


# SOM聚类算法
def som_clustering(datas):
    print("当前聚类算法选择SOM自组织映射神经网络")
    SOM = som.CyrusSOM(epochs=5)
    labs = SOM.transform_fit(datas)
    raito = 0
    n_clusters = len(set(labs)) - (1 if -1 in labs else 0)  # 获取分簇的数目
    return labs, n_clusters, raito


"""
所有后缀为true的聚类算法均表示，使用带有真实标签的ARI系数方法选取最优参数
"""
# 使用ARI系数选参的k-means算法
def k_means_true(datas, true_labs):
    print("当前使用带有真实标签的k-means算法")
    best_k = []
    N = 10  # 最多分十类
    for n in range(2, N + 1):  # 从2-10中寻找最佳k值
        km = KMeans(n_clusters=n, init='k-means++', random_state=0).fit(datas)
        pred_labs = km.labels_
        evaluate_res = metrics.adjusted_rand_score(true_labs, pred_labs)  # 计算ARI系数
        best_k.append((evaluate_res, n, pred_labs))

    best_k = sorted(best_k, key=lambda x: x[0])[-1]
    evaluate_res, n_clusters, pred_labs = best_k
    raito = len(pred_labs[pred_labs[:] == -1]) / len(pred_labs)  # 计算噪声点个数占总数的比例
    return pred_labs, n_clusters, raito


# 使用ARI选参的层次聚类方法
def hierarchical_clustering_true(datas, true_labs):
    print("当前使用带有真实标签的层次聚类算法")
    best_n = []
    N = 10  # 最多分十类
    for n in range(2, N + 1):  # 从2-10中寻找最佳聚簇
        model = AgglomerativeClustering(n_clusters=n, linkage='ward').fit(datas)
        pred_labs = model.labels_
        evaluate_res = metrics.adjusted_rand_score(true_labs, pred_labs)
        best_n.append((evaluate_res, n, pred_labs))

    best_n = sorted(best_n, key=lambda x: x[0])[-1]
    evaluate_res, n_clusters, pred_labs = best_n
    raito = len(pred_labs[pred_labs[:] == -1]) / len(pred_labs)  # 计算噪声点个数占总数的比例
    return pred_labs, n_clusters, raito


# 使用ARI选参的高斯混合模型方法
def gaussian_mixture_true(datas, true_labs):
    print("当前使用带有真实标签的GMM高斯混合模型")
    best_n = []
    N = 10  # 最多分十类
    for n in range(2, N + 1):  # 从2-10中寻找最佳聚簇
        model = GaussianMixture(n_components=n)
        pred_labs = model.fit_predict(datas)
        evaluate_res = metrics.adjusted_rand_score(true_labs, pred_labs)
        best_n.append((evaluate_res, n, pred_labs))

    best_n = sorted(best_n, key=lambda x: x[0])[-1]
    evaluate_res, n_clusters, pred_labs = best_n
    raito = len(pred_labs[pred_labs[:] == -1]) / len(pred_labs)  # 计算噪声点个数占总数的比例
    return pred_labs, n_clusters, raito


# 使用ARI选参的谱聚类方法
def spectral_clustering_true(datas, true_labs):
    print("当前使用带有真实标签的谱聚类算法")
    best_n = []
    N = 10  # 最多分十类
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        for n in range(2, N + 1):
            pred_labs = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(datas)
            evaluate_res = metrics.adjusted_rand_score(true_labs, pred_labs)
            best_n.append((evaluate_res, n, pred_labs, gamma))

    best_n = sorted(best_n, key=lambda x: x[0])[-1]
    evaluate_res, n_clusters, pred_labs, gamma = best_n
    raito = len(pred_labs[pred_labs[:] == -1]) / len(pred_labs)  # 计算噪声点个数占总数的比例
    return pred_labs, n_clusters, raito


# 使用ARI系数选参的SOM聚类算法
def som_clustering_true(datas, true_labs):
    print("当前使用带有真实标签的SOM自组织神经网络算法")
    best_n = []
    for index, epochs in enumerate((5, 10, 15, 20)):
        SOM = som.CyrusSOM(epochs=epochs)
        pred_labs = SOM.transform_fit(datas)
        evaluate_res = metrics.adjusted_rand_score(true_labs, pred_labs)
        best_n.append((evaluate_res, pred_labs,epochs))

    best_n = sorted(best_n, key=lambda x: x[0])[-1]
    evaluate_res, pred_labs, epochs = best_n
    n_clusters = len(set(pred_labs)) - (1 if -1 in pred_labs else 0)  # 获取分簇的数目
    raito = 0
    return pred_labs, n_clusters, raito
