# -*- coding: utf-8 -*-

import numpy as np
import pickle
import torch
from nltk.corpus import framenet as fn


# 加载训练数据
def load_data(fn_lu_embedding_filename, fn_plus_lu_embedding_filename, num_dict):
    fn_E = pickle.load(open(fn_lu_embedding_filename, 'rb'))
    fn_plus_E = pickle.load(open(fn_plus_lu_embedding_filename, 'rb'))
    types_to_int = {
        '公司上市': 0, '股东减持': 1, '股东增持': 2, '企业收购': 3, '企业融资': 4,
        '股份回购': 5, '股权质押': 6, '质押': 6, '解除质押': 7, '企业破产': 8,
        '亏损': 9, '被约谈': 10, '中标': 11, '高管变动': 12, '股权冻结': 13
    }

    X = list()
    Y = list()

    for text_id, tensor in fn_E.items():
        event_type = num_dict[str(text_id)]
        # # 需要用于dec2vec的数据加载
        # tensor = torch.from_numpy(tensor)
        X.append(tensor)
        Y.append(types_to_int[event_type])

    cut_off = len(Y)

    for keys, tensor in fn_plus_E.items():
        event_type, triple_data = keys
        # # 需要用于dec2vec的数据加载
        # tensor = torch.from_numpy(tensor)
        X.append(tensor)
        Y.append(types_to_int[event_type])

    return np.array(X), np.array(Y), len(types_to_int), cut_off

# 加载异常数据集
def load_anomalous_synsets(anomalous_synset_embedding_filename):
    W = pickle.load(open(anomalous_synset_embedding_filename, 'rb'))
    anom_X = list()
    anom_Y = list()

    for synset_name, tensor in W.items():
        # tensor = torch.from_numpy(tensor)
        anom_X.append(tensor)
        anom_Y.append(synset_name)

    pickle.dump([np.array(anom_X), np.array(anom_Y)], open("corrupted_anom.p", 'wb'))
    return np.array(anom_X), np.array(anom_Y)
