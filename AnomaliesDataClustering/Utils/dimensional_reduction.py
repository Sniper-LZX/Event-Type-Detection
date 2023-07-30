from sklearn.manifold import TSNE
from sklearn import decomposition
import umap


# 使用PCA算法降维
def dr_pca(tensors, dimension):
    pca = decomposition.PCA(n_components=dimension)
    tensors_embedded = pca.fit_transform(tensors)
    # print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（方差大小）
    # print(pca.explained_variance_ratio_)  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
    # print(pca.explained_variance_ratio_.sum())  # 总占比
    return tensors_embedded


# PCA按信息量进行降维
# tensors: 原数据; percentage:信息量比例
def dr_pca_information_content(tensors, percentage):
    pca_f = decomposition.PCA(n_components=percentage, svd_solver="full")
    x_f = pca_f.fit_transform(tensors)
    return x_f


# 使用t-SNE进行降维
def dr_tSNE(tensors, dimension):
    # random_state:固定一个随机种子，使得每次压缩后的向量都保持一致
    # method:表示两种优化方法。method=barnets_hut，耗时O(NlogN)；method=exact耗时O(N^2)但是误差小，同时不能用于百万级样本
    # early_exaggeration:表示嵌入空间簇间距的大小，默认为12，该值越大，可视化后的簇间距越大
    # n_iter:迭代次数，默认为1000，自定义设置时应保证大于250
    tensors_embedded = TSNE(n_components=dimension, init='pca', random_state=10).fit_transform(tensors)
    # print(tensors_embedded)
    return tensors_embedded


# 使用NMF进行降维,要求所有样本数据非负
def dr_nmf(tensors, dimension):
    nmf = decomposition.NMF(n_components=dimension)
    # 训练模型
    nmf.fit(tensors)
    # 降维
    nmf_features = nmf.transform(tensors)
    return nmf_features


# 使用UMAP算法进行降维
def dr_umap(tensors, dimension):
    # UMAP算法参数
    # n_neighbors：确定相邻点的数量; 值越大代表更多的全局结构被保留，而失去了详细的局部结构
    # 一般来说，这个参数应该在5到50之间，10到15是一个合理的默认值。
    # min_dist：控制允许嵌入的紧密程度
    # 数值越大，嵌入点分布越均匀;数值越小，算法对局部结构的优化越精确。合理的值在0.001到0.5之间，0.1是合理的默认值。
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=40, n_components=dimension)
    umap_data = umap_model.fit_transform(tensors)
    return umap_data

