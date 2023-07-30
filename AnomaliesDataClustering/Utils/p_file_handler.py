import numpy as np
# try:
#     import cPickle
# except BaseException:
#     import _pickle as cPickle

import _pickle as cPickle


# 处理p文件，分别获取其三元簇集合与tensor向量集合
# path : p文件所在系统路径，绝对路径相对路径均可
def file_handler(path):
    # step 1
    # save npy data
    ano_data = cPickle.load(open(path, 'rb'))
    save_path = path.replace(".p", ".npy")
    np.save(save_path, ano_data)

    # step 2
    # 将npy data 转化为字典数据
    mat = np.load(save_path, allow_pickle=True)
    ano_dict = mat.tolist()

    # 将所有键的集合转化为一个列表
    all_keys = list(ano_dict.keys())

    all_values = list(ano_dict.values())
    for i in range(len(all_values)):
        if i == 0:
            ano_tensors = all_values[i].numpy()  # 获取Bert嵌入数据
            # ano_tensors = all_values[i]  # 获取Doc2vec嵌入数据
        else:
            ano_tensors = np.row_stack((ano_tensors, all_values[i]))

    return all_keys, ano_tensors


# 处理p文件，只返回其tensor向量
# path : p文件所在系统路径，绝对路径相对路径均可
def file_handler_tensor(path):
    # step 1
    # save npy data
    ano_data = cPickle.load(open(path, 'rb'))
    save_path = path.replace(".p", ".npy")
    np.save(save_path, ano_data)

    # step 2
    # 将npy data 转化为字典数据
    mat = np.load(save_path, allow_pickle=True)
    ano_dict = mat.tolist()

    all_values = list(ano_dict.values())
    for i in range(len(all_values)):
        if i == 0:
            ano_tensors = all_values[i].numpy()  # 获取Bert嵌入数据
            # ano_tensors = all_values[i]  # 获取Doc2vec嵌入数据
        else:
            ano_tensors = np.row_stack((ano_tensors, all_values[i]))

    return ano_tensors


# 根据npy文件获取向量数据
# 适用于已经处理过p文件进而直接利用其生成npy文件的情况，不重复处理p文件
def file_handler_npy(npy_path):

    mat = np.load(npy_path, allow_pickle=True)
    ano_dict = mat.tolist()

    all_values = list(ano_dict.values())
    for i in range(len(all_values)):
        if i == 0:
            ano_tensors = all_values[i].numpy()  # 获取Bert嵌入数据
            # ano_tensors = all_values[i]  # 获取Doc2vec嵌入数据
        else:
            ano_tensors = np.row_stack((ano_tensors, all_values[i]))

    return ano_tensors




