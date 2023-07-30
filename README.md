# 金融事件类型增量标注方法的研究与实现


## 数据准备

我们所使用的数据集如下
1. 来自百度千言的DuEE-fin
2. ChFinAnn，来自Doc2EDAG
3. 阿里和蚂蚁在CCKS上的数据集

## 各个文件夹执行的顺序

### Bert_emded 文件夹
- 功能：将数据进行Bert嵌入
- 执行Python脚本：`bert_embed.py`
- 输入数据：`Training_Data`
- 输出数据：`Embed_Data`
- 通用模型：`chinese_L-12_H-768_A-12`

### doc2vec 文件夹
- 功能：将数据进行doc2vec嵌入
- 原始数据处理（使得数据以 `.csv` 的格式存储，使数据能够进行嵌入）
    - 执行 Python 脚本：`buildcsv.py`
    - 输入数据：`Training_Data`
    - 输出数据: `data`
- doc2vec嵌入
    - 执行 Python 脚本：`doc2vec.py`
    - 输入数据：`data` 和 `Training_Data`
    - 输出数据：`Vectors`

### SDEC-AD-my 文件夹

#### 数据准备
文件夹 `mydata/` 包含了所有的原始数据，以及已经嵌入得到的所有数据包括四个文件夹：
- 用bert嵌入原文的数据（Bert_Text）
- 用bert嵌入三元组簇的数据（Bert_Triple）
- 用doc2vec嵌入原文的数据（doc2vec_text）
- 用doc2vec嵌入三元组簇的数据（doc2vec_triple）

#### 半监督深度聚类
1. 用文件夹`trained_SDEC_AD/`来保存训练好的权重。
2. 运行 Python 脚本`python event_types_trian.py`来训练 SDEC-AD 模型。
3. 运行 Python 脚本 `python event_types_pred.py` 来预测和评估聚类事件类型数据的结果. 将脚本中的参数 `SDEC_trained_weights` 更新为具有最大 Bcubed F1 分数的训练权重

#### 异常事件检测
1. 按照上一节“半监督深度聚类”中的说明生成经过训练的权重。
2. 更新 Python 脚本 `anomaly_detection.py` 中的参数 `SDEC_trained_weights` 为具有最大 Bcubed F1 分数的训练权重。
 然后，运行 Python 脚本 `anomaly_detection.py` 来训练解码器并检测异常事件类型。
