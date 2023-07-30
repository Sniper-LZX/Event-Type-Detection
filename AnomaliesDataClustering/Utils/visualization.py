import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata

UNCLASSIFIED = 0
NOISE = -1


# 绘制二维聚类结果图
def draw_cluster2d(datas, labs, n_cluster):
    plt.cla()

    # 设置聚类点颜色
    colors = colors_no_yellow(n_cluster)

    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(datas[i ,0] ,datas[i ,1], s=16., color=(0, 0, 0))
        else:
            plt.scatter(datas[i, 0], datas[i, 1], s=16., color=colors[lab])
    plt.savefig("../data/word_cloud/cluster_distribution.jpg")
    plt.show()


# 绘制三维聚类结果图
def draw_cluster3d(datas, labs, n_cluster):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # 设置聚类点颜色
    colors = colors_no_yellow(n_cluster)

    for i, lab in enumerate(labs):
        if lab == NOISE:
            ax.scatter(datas[i, 0], datas[i, 1], datas[i, 2], zdir="z", s=20, color=(0, 0, 0), marker="o")
        else:
            ax.scatter(datas[i, 0], datas[i, 1], datas[i, 2], zdir="z", s=20, color=colors[lab - 1], marker="o")

    ax.set(xlabel="X", ylabel="Y", zlabel="Z")

    plt.show()


# 程序要聚多少种类，就设置多少种颜色,包含浅黄色
def colors_contain_yellow(n_cluster):
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, n_cluster)]
    return colors


# 暴力设置聚类颜色，不包含浅黄色
    # 开始黄色偏见,头两个颜色单纯因为喜欢
def colors_no_yellow(n_cluster):
    # 'e50000':红色; '#c79fef':淡紫色; '#f97306':橘色; '#029386':茶色; '#75bbfd':天蓝色;
    # '#ae7181':肉桂色; '#ff028d':亮粉色; '#ad8150':亮棕色; '#ff796c':浅橙色; '#15b01a': 绿色; '#0343df':蓝色
    colors = ['#e50000', '#c79fef', '#f97306', '#029386', '#75bbfd', '#ae7181', '#ff028d',
              '#ad8150', '#ff796c', '#15b01a', '#0343df', '#DC143C', '#FFB6C1', '#DB7093',
              '#C71585', '#8B008B', '#4B0082', '#7B68EE', '#0000FF', '#B0C4DE', '#708090',
              '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA', '#008000', '#FFFF00', '#808000', ]

    if n_cluster <= len(colors):
        return colors
    else:
        print("黄色入侵，聚那么多类，咋不蠢死你")
        return colors_contain_yellow(n_cluster)


# 指定一个函数用于计算每个点的高度，也可以直接使用二维数组储存每个点的高度
def height_computation(x, y):
    return (1 - y ** 5 + x ** 5) * np.exp(-x ** 2 - y ** 2)


# 利用核函数估计，绘制密度等高线图
def kernel_density_contour(datas):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    x = datas[:, 0]
    y = datas[:, 1]
    # 空间坐标网格化
    x, y = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))

    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(datas)  # 核密度估计
    s = kde.score_samples(datas)  # 数据点核密度

    # 通过griddata函数对核密度z进行插值
    zz = griddata(points=datas, values=s, xi=(x, y), method="linear")  # 对纵坐标也就是核密度进行插值
    # 应用contourf绘制等高线图
    a = np.linspace(np.nanmin(zz), np.nanmax(zz), 10)
    myc = plt.contourf(x, y, zz, levels=a, alpha=0.75, cmap='Blues')  # cmap=plt.cm.hot，可以取Blues
    cbar = plt.colorbar(myc)

    plt.title('密度等高线图')
    plt.show()


# 绘制二维散点图
def draw_2d_graph(datas, color_lab, colors_number):
    plt.cla()

    # 设置散点图共用多少种颜色表示
    colors = colors_no_yellow(colors_number)

    for i, data in enumerate(datas):
        plt.scatter(data[0], data[1], s=2000., color=colors[color_lab])

    plt.axis('off')
    plt.savefig("../data/word_cloud/cluster"+ str(color_lab) + ".jpg")
    # plt.show()

