# SDEC-AD for Semantic Frame Induction

## 用法

#### 依赖库
依赖环境
- bcubed==1.5
- nltk==3.4.3
- matplotlib==3.2.1
- numpy==1.18.2
- Keras==2.2.5
- scikit_learn==0.23.0
- tensorflow==1.15.0

可以简单运行 `pip3 install -r requirements.txt` 安装环境

### 数据准备
我们所使用的数据集如下
1. 来自百度千言的DuEE-fin
2. ChFinAnn，来自Doc2EDAG
3. 阿里和蚂蚁在CCKS上的数据集

文件夹 `mydata/` 包含了所有的原始数据，以及已经嵌入得到的所有数据包括四个文件夹：
- 用bert嵌入原文的数据（Bert_Text）
- 用bert嵌入三元组簇的数据（Bert_Triple）
- 用doc2vec嵌入原文的数据（doc2vec_text）
- 用doc2vec嵌入三元组簇的数据（doc2vec_triple）

### 半监督深度聚类
1. 用文件夹`trained_SDEC_AD/`来保存训练好的权重。
2. 运行 Python 脚本`python event_types_trian.py`来训练 SDEC-AD 模型。
3. 运行 Python 脚本 `python event_types_pred.py` 来预测和评估聚类事件类型数据的结果. 将脚本中的参数 `SDEC_trained_weights` 更新为具有最大 Bcubed F1 分数的训练权重

### 异常事件检测
1. 按照上一节“半监督深度聚类”中的说明生成经过训练的权重。
2. 更新 Python 脚本 `python anomaly_detection.py` 中的参数 `SDEC_trained_weights` 为具有最大 Bcubed F1 分数的训练权重。
 然后，运行 Python 脚本 `python anomaly_detection.py` 来训练解码器并检测异常事件类型。

### 利用重构误差区分正常事件和异常事件
运行 Python 脚本 `python Bicluster.py` 
