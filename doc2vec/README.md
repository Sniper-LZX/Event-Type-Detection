# Doc2Vec

## 用法

### 依赖库
依赖python环境
- pandas==0.23.4
- gensim==3.6.0
- numpy==1.16.0
- scikit_learn==0.20.2

可以简单运行 `pip3 install -r requirements.txt` 安装环境

### 执行步骤
1. 运行 Python 脚本 `python buildcsv.py` 来生成doc2vec所需要的 `.csv` 数据，存放在 `data` 文件夹中
2. 运行 Python 脚本 `python doc2vec.py` 对数据进行doc2vec向量嵌入，生成的数据存放在 `Vectors` 文件夹中