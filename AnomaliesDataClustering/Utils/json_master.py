import json


# 根据指定属性，以列表的形式返回json特定数据
# attribute可取值：Triplet, event_type, text
def json_reader_specific_values(file_path, attribute):
    value_list = []
    with open(file_path, 'r', encoding='utf-8') as af:
        k = 0
        for line in af.readlines():
            k += 1
            # print('第', k, '条text')  # 识别进度
            text = json.loads(line)
            content = text[attribute]
            value_list.append(content)
    return value_list
