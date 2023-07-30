from bert_serving.client import BertClient
import json
import torch
import pickle

"""
已经处理过的数据，包括：
1、去除事件类型为None的信息
2、去除事件类型为多种不同类型的信息
3、去除三元组抽取为空的信息，即"Triplet" : "None"
4、去除triplet长度（第一个文件）和triplet长度+event_type长度+1（第二个文件）超过512的信息
        # ‘+1’是因为事件类型和三元组使用逗号隔开，如：质押，<三元组簇>，其中逗号即为那个‘+1’
5、已经将多个相同事件类型简化为一个事件类型，赋给event_type

"""


first_file = 'Training_Data/train_first_embed.json'
second_file = 'Training_Data/train_second_embed.json'
anomalies_file = 'Training_Data/train_anomalies_embed.json'
# # 13种事件类型
# event_type = ["公司上市", "股东减持", "股东增持", "企业收购", "企业融资", "股份回购", "股票质押",
#               "解除质押", "企业破产", "亏损", "被约谈", "中标", "高管变动"]

bc = BertClient()

triple_to_int = {}
triple_to_event = {}
text_num_type = {}

dict_one_text = {}
dict_one_triple = {}
dict_two_text = {}
dict_two_triple = {}
dict_thr_text = {}
dict_thr_triple = {}

print('处理第一组数据')
with open(first_file, 'r', encoding='utf-8') as f_first:
    k = 0       # 总共多少条
    for line in f_first.readlines():
        k += 1
        print('第', k, '条item')
        # 一个实例
        item = json.loads(line)

        E_type = item["event_type"]                             # 事件类型
        Order_n = item["Order_number"]                          # 实例标号
        E_triple = str(item["Triplet"])[1:-1]                   # 三元组簇
        Text = item["text"]
        if isinstance(Text, list):
            Text = ''.join(Text)
        if len(Text) > 512:
            Text = Text[:512]

        text_num_type[item["Order_number"]] = E_type          # 形成字典，{每条ID：事件类型, ......}

        dict_one_triple[Order_n] = torch.tensor(bc.encode([E_triple])[0])
        dict_one_text[Order_n] = torch.tensor(bc.encode([Text])[0])
    f_first.close()

# 三元组和原文保存成 p 类型文件
pic1 = open('Embed_Data/Triple/triple_bert_first.p', 'wb')
pic2 = open('Embed_Data/Text/text_bert_first.p', 'wb')
pickle.dump(dict_one_triple, pic1)
pickle.dump(dict_one_text, pic2)
pic1.close()
pic2.close()

# 类型字典 1--------------------------------------------------------------
with open('text_num_type1.json', 'w', encoding='utf-8') as t_n_t:
    data1 = json.dumps(text_num_type, ensure_ascii=False)
    t_n_t.write(data1)
t_n_t.close()
# -------------------------------------------------------------------------

print("处理第二组数据")
with open(second_file, 'r', encoding='utf-8') as f_second:
    k = 0       # 总共多少条
    for line in f_second.readlines():
        k += 1
        print('第', k, '条item')

        item = json.loads(line)

        # 8608是第一个文件中项目的总数，需要追加字典
        text_num_type[item["Order_number"]+8608] = E_type

        E_type = item["event_type"]                     # 事件类型
        Order_n = item["Order_number"]                  # 实例标号
        E_triple = item["Triplet"]                      # 三元组簇
        Text = item["text"]                             # 文本原文
        if isinstance(Text, list):
            Text = ''.join(Text)
        if len(Text) > 512:
            Text = Text[:512]

        if len(Text) > 23:
            tag_text = '{} {}{}'.format(Order_n, Text[:20], '...')
        else:
            tag_text = '{} {}'.format(Order_n, Text)
        Text_tuple = (E_type, tag_text)

        if len(E_triple) > 23:
            tag_triple = '{} {}{}'.format(Order_n, E_triple[:20], '...')
        else:
            tag_triple = '{} {}'.format(Order_n, E_triple)
        Triple_tuple = (E_type, tag_triple)

        dict_two_text[Text_tuple] = torch.tensor(bc.encode([Text])[0])
        dict_two_triple[Triple_tuple] = torch.tensor(bc.encode([E_triple])[0])
    f_second.close()

# 三元组和原文保存成 p 类型文件
pic3 = open('Embed_Data/Triple/triple_bert_second.p', 'wb')
pic4 = open('Embed_Data/Text/text_bert_second.p', 'wb')
pickle.dump(dict_two_triple, pic3)
pickle.dump(dict_two_text, pic4)
pic3.close()
pic4.close()
# 类型字典 2--------------------------------------------------------------
with open('text_num_type2.json', 'w', encoding='utf-8') as t_n_t:
    data1 = json.dumps(text_num_type, ensure_ascii=False)
    t_n_t.write(data1)
t_n_t.close()
# -------------------------------------------------------------------------

print('处理异常数据')
# 异常事件的数据嵌入
with open(anomalies_file, 'r', encoding='utf-8') as af:
    k = 0
    for line in af.readlines():
        k += 1
        print('第', k, '条text')

        item = json.loads(line)
        E_triple = str(item["Triplet"])[1:-1]           # 三元组簇
        Text = item["text"]                             # 文本原文
        Order_n = item["Order_number"]                  # 实例标号

        if isinstance(Text, list):                      # 处理列表式原文
            Text = ''.join(Text)
        if len(Text) > 512:
            Text = Text[:512]

        tag_text = (Order_n, Text)
        tag_triple = (Order_n, E_triple)

        dict_thr_text[tag_text] = torch.tensor(bc.encode([Text])[0])
        dict_thr_triple[tag_triple] = torch.tensor(bc.encode([E_triple])[0])

        print(len(dict_thr_text), len(dict_thr_triple))

pic5 = open('Embed_Data/Text/text_bert_anomalies.p', 'wb')
pic6 = open('Embed_Data/Triple/triple_bert_anomalies.p', 'wb')
pickle.dump(dict_thr_text, pic5)
pickle.dump(dict_thr_triple, pic6)
pic5.close()
pic6.close()
