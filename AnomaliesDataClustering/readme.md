# 一种基于异常检测的金融事件类型增量发现方法

## 依赖

​		本实验所需环境依赖如下：

```python
python == 3.7
numpy == 1.18.2
matplotlib==3.2.1
scikit_learn==0.23.0
gensim == 3.8.3
wordcloud == 1.8.2.2
```

## 数据集

​		本实验所用数据集存放在data目录下，各个数据文件介绍如下：

```python
bert_text: 用bert嵌入异常事件文本所得到的结果
bert_triplet: 用bert嵌入异常事件三元组所得到的结果
doc_text: 用doc2vec嵌入异常事件文本所得到的结果
doc_triplet:用doc2vec嵌入异常事件三元组所得到的结果
json: 存放本实验所有正常已知事件与异常未知事件的json文件集合
      train_anomalies_embed.json代表所有异常未知事件
keywords: 存放本实验在进行关键词提取时所用到的停用词信息
record: 记录实验结果
```

​		代码可以通过Utils->p_file_handler，解析不同数据文件，进而获得不同嵌入数据，从而进行消融实验。

## 代码结构介绍

​		本实验各部分代码的功能如下：

​		Cluster->clustering_master:存放聚类实验所需各项聚类算法。

​		keyword-extraction->LDA_model:存放关键字提取算法

​		Utils:

- dimensional_reduction:  降维
- distance_metric: 计算向量距离
- evaluation: 聚类效果评估
- json_master:  处理json文件
- normalization: 归一化
- p_file_handler: 处理嵌入p文件
- visualization:可视化
- word_cloud:关键词词云生成