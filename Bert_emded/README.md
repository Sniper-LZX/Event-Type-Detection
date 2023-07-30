# 对数据进行bert嵌入
## 用法
#### 依赖Python库
安装服务端和客户端
- bert-serving-server
- bert-serving-client

运行 `pip3 install bert-serving-server` 和 `pip3 install bert-serving-client` 安装依赖库。

#### 数据
- 文件夹 `Training_Data` 包含准备用来进行嵌入的原始数据
- 文件夹 `Embed_Data` 存储已经嵌入的数据

#### 使用步骤
1. 在 https://github.com/google-research/bert/ 下载预训练模型 `chinese_L-12_H-768_A-12`
2. 在cmd命令中转入python库的安装路径(Scripts)，使用命令来启动bert服务器并指定训练文本的长度，最长512
   `bert-serving-start -model_dir <模型路径> -seq_max_len 512`
3. 运行 Python 脚本 `python bert_embed.py` 生成三组bert嵌入的数据

#### 输出
- bert嵌入的数据集
- 数据集1和数据集的