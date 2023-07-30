from Cluster import clustering_master
from keyword_extraction import LDA_model as lda

from Utils import visualization as vs
from Utils import p_file_handler as handler
from Utils import dimensional_reduction as dr
from Utils import normalization as nz
from Utils import evaluation

from sklearn.preprocessing import StandardScaler
import numpy as np
from Utils import json_master as jm
import codecs


if __name__ == "__main__":
    """
    step 1
    获取异常事件的高斯混合模型聚类结果，
    聚类实验使用数据本身的真实标签进行辅助聚类，以得到实验阶段所能获得的最优效果。
    """

    # 提取未知异常数据本身的真实事件类型标签
    json_path = 'data/json/train_anomalies_embed.json'
    true_labs = jm.json_reader_specific_values(json_path, "event_type")

    # 获取全部的事件类型集合
    lab_set = []
    for lab in true_labs:
        if lab not in lab_set:
            lab_set.append(lab)
    print(lab_set)

    # 将事件类型标签映射为阿拉伯数字，方便计算聚类评价指标
    i = 0
    lab_to_num = {}
    for lab in lab_set:
        lab_to_num[lab] = i
        i += 1

    i = 0
    for lab in true_labs:
        true_labs[i] = lab_to_num[lab]
        i += 1
    print(true_labs)
    print(len(true_labs))

    tripet_path = 'data/bert_triplet/triple_bert_anomalies.p'
    triplet_set, tripet_tensors = handler.file_handler(tripet_path)

    # 数据正则化
    tripet_tensors = StandardScaler().fit_transform(tripet_tensors)

    # 将高维数据降至两维，便于可视化
    tensor_embedding = dr.dr_tSNE(tripet_tensors, 2)

    # 数据归一化，消除数据量纲影响，提高聚类评价指标准确性
    # 避免在计算向量间距离的过程中，某些向量在某些维度取值过大，影响聚类效果
    tensor_embedding = nz.z_score_normalization(tensor_embedding)

    # 使用聚类算法:高斯混合模型
    # 采用真实标签辅助聚类，获取高斯混合模型实验阶段最优的聚类性能
    labs, n_clusters, raito = clustering_master.gaussian_mixture(tensor_embedding)
    # labs, n_clusters, raito = clustering_master.gaussian_mixture(tensor_embedding)

    # print("各点类别标签：", labs)
    print("聚簇数量：", n_clusters)
    print("噪声点所占比例", raito)

    # 打印无监督评价指标
    print("轮廓系数分数为：", evaluation.sc_score(tensor_embedding, labs))
    print("CH分数为：", evaluation.ch_score(tensor_embedding, labs))
    print("DBI取值为：", evaluation.dbi_score(tensor_embedding, labs))

    # # 绘制可视化图形
    # vs.draw_cluster2d(tensor_embedding, labs, n_clusters)
    # # vs.draw_cluster3d(tensor_embedding, labs, n_clusters)
    #
    # # 绘制密度等高线图
    # vs.kernel_density_contour(tensor_embedding)

    # """
    # step 2:使用原文进行关键词提取
    # 获取异常事件的全部文本信息。
    # """
    # # 从json文件中提取所有异常事件的文本信息
    # text_set = jm.json_reader_specific_values(json_path, "text")
    #
    # # 设置文本所应屏蔽的常用非关键词集合
    # replace_dict = ["证券代码", "证券简称", "股票简称", "股票代码", "公告编号", "编号", "&nbsp", "主办券商",
    #                 "简称", "（A股）", "（B股）", "编码",
    #                 "本公司董事会及全体董事保证", "管理人保证", "本公告内容", "公告内容的", "本报告所载资料", "真实", "准确", "完整",
    #                 "内容", "虚假记载", "虚假记载", "重大遗漏", "负连带责任", "不存在任何", "并对其内容的", "特别提示",
    #                 "真实性、准确性和完整性", "承担个别及连带责任", "本公司及全体董事、监事、高级管理人员保证", "本公司董事会及其董事保证",
    #                 "本公司及董事会全体成员保证", "误导性陈述", "对公告的", "或者", "并对其", "性", "对公告中的"]
    #
    # for i in range(len(text_set)):
    #     if isinstance(text_set[i], str):
    #         for rp in replace_dict:
    #             text_set[i] = text_set[i].replace(rp, "")
    #     else:
    #         text_set[i] = "".join(text_set[i])
    #         for rp in replace_dict:
    #             text_set[i] = text_set[i].replace(rp, "")
    #
    # """
    # step 3
    # 按聚类结果将所有被聚为一类的文本融合在一起，便于关键词提取。
    # """
    # # 有多少种聚类结果，就生成多大的空列表
    # text_cluster = []
    # for i in range(n_clusters):
    #     text_cluster.append("")
    #
    # # 按聚类结果将不同文本归类到一起
    # for i in range(len(labs)):
    #     text_cluster[labs[i]] = text_cluster[labs[i]] + text_set[i]
    #
    # for item in range(len(text_cluster)):
    #     print("第" + str(item) + "类的关键词提取结果： ")
    #     lda.keywords_extraction(text_cluster[item])

    """
    step 2: 使用三元组簇进行关键词提取
    获取异常事件所有的三元组簇形成一个三元簇集合，并将其处理成LDA模型所需要的数据格式
    """
    # triplet_set是一个三维列表，其中每个二维列表代表一个事件的所有三元组簇，格式为[[...], [...], [...], ...]
    triplet_set = jm.json_reader_specific_values(json_path, "Triplet")

    # 为满足LDA模型输入要求
    # 将每个异常事件的三元组簇由原先的[[...], [...], [...], ...]
    # 合并成[[..., ..., ...]]
    for triplet in triplet_set:
        for i in range(1, len(triplet)):
            triplet[0].extend(triplet[1])
            triplet.pop(1)

    # 设置关键词提取的停用词（不可被当成的关键词的常用词）
    stopwords = [line.strip() for line in
                 codecs.open(r'data/keywords/stopwords_triplet.txt', 'r', 'utf-8').readlines()]
    # 设置无意义词列表（三元组中元素一旦包含这些词，则不再参加关键词提取）
    meaning_less = ["证券代码", "证券简称", "股票简称", "股票代码", "公告编号", "编号", "&nbsp", "主办券商",
                    "简称", "（A股）", "（B股）", "编码", "有限公司", "本公司", "证券", "股份有限公司", "注意", "有限责任公司",
                    "本公司董事会及全体董事保证", "管理人保证", "本公告内容", "公告内容的", "本报告所载资料", "真实", "准确", "完整",
                    "内容", "虚假记载", "虚假记载", "重大遗漏", "负连带责任", "不存在任何", "并对其内容的", "特别提示",
                    "真实性、准确性和完整性", "承担个别及连带责任", "本公司及全体董事、监事、高级管理人员保证", "本公司董事会及其董事保证",
                    "本公司及董事会全体成员保证", "误导性陈述", "对公告的", "或者", "并对其", "性", "对公告中的"]

    # 去除每个事件所代表的三元簇中的停用词以及包含无意义词的元素
    for triplet in triplet_set:
        purification = []
        for i in range(len(triplet[0])):
            if triplet[0][i] not in stopwords and triplet[0][i] != "":
                flag = 1  # 每次迭代都更新flag的值为1
                # 如果当前三元组中实体元素包含任一meaning_less中的无意义单词，则设置flag=0
                for word in meaning_less:
                    if word in triplet[0][i]:
                        flag = 0
                        break
                    # 如果当前三元组实体即不属于stopwords停用词也不为空
                    # 同时也不包含任何需要被消除的无意义词则将其加入到提纯列表purification中
                if flag != 0:
                    purification.append(triplet[0][i])
        triplet[0] = purification  # 用提纯后的三元组簇列表替换原先的列表

    # for item in triplet_set:
    #     print("*******:", item)

    # 有多少个聚类结果就设置多少个二维列表[[]],
    # 将所有二维列表组合为一个大的三维列表用于存放所有聚类簇的三元组簇集合。
    # triplet_cluster格式：[[[第0号聚类簇所对应三元组簇集合]], [[第1号聚类簇所对应三元组簇集合]], ...]
    # 每个二维列表的格式：[[某一聚类簇全部的三元组实体元素]]
    triplet_cluster = []
    for i in range(n_clusters):
        triplet_cluster.append([[]])

    # 按聚类结果将不同事件所代表的三元组实体元素二维列表归类到一起
    # 假设[[a]]与[[b]]被聚类为一类且其聚类标号为1，
    # 则本代码会将其都存入triplet_cluster中的第1号聚类簇对应位置，即:[[[]], [[a,b]], ...](聚类簇编号从0开始)，以此类推
    # a=[[1,2]] b=[[3, 4]],执行a[0].extend(b[0])后，a=[[1,2,3,4]]
    for i in range(len(labs)):
        triplet_cluster[labs[i]][0].extend(triplet_set[i][0])

    """
    step 3: 使用三元组簇进行关键词提取
    针对每一个聚类簇的所有三元组簇集合进行关键词提取
    """
    # 对每一类聚类结果进行关键词提取
    words_list = []
    for i in range(n_clusters):
        words_list.append([])

    for i in range(len(triplet_cluster)):
        print("第" + str(i) + "类的关键词提取结果： ")
        model = lda.LDA_model(triplet_cluster[i])
        keywords = model.show_topic(0, 10)
        # print('输出该主题的的词及其词的权重:')
        print(keywords)
        for item in keywords:
            words_list[i].append(item[0])

    print(words_list)

    with open("data/keywords/extract", "w+") as f:
        f.write(str(words_list))















