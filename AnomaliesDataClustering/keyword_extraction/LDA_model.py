from gensim import corpora, models

import jieba
import wordcloud
import matplotlib.pyplot as plt

import codecs


# 简单文本处理
def get_text(texts):
    #     flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    #     stopwords = ('的', '就', '是', '用', '还', '在', '上', '作为','为','被')
    stopwords = [line.strip() for line in
                 codecs.open(r'data/keywords/stopwords.txt', 'r', 'utf-8').readlines()]
    jieba.load_userdict(r'data/keywords/financial_corpus.txt')

    words_list = []
    line_word = []
    words = jieba.cut(str(texts), HMM=True)
    # 去停用词
    for word in words:
        if word not in stopwords and word != " ":
            #         words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
            line_word.append(word)
    words_list.append(line_word)

    return words_list


# 生成LDA模型
def LDA_model(words_list):
    # 构造词典
    # Dictionary()方法遍历所有的文本，为每个不重复的单词分配一个单独的整数ID，同时收集该单词出现次数以及相关的统计信息
    dictionary = corpora.Dictionary(words_list)
    #     print(dictionary)
    #     print('打印查看每个单词的id:')
    #     print(dictionary.token2id)  # 打印查看每个单词的id

    # 将dictionary转化为一个词袋
    # doc2bow()方法将dictionary转化为一个词袋。得到的结果corpus是一个向量的列表，向量的个数就是文档数。
    # 在每个文档向量中都包含一系列元组,元组的形式是（单词 ID，词频）
    corpus = [dictionary.doc2bow(words) for words in words_list]
    #     print('输出每个文档的向量:')
    #     print(corpus)  # 输出每个文档的向量

    # LDA主题模型
    # num_topics -- 必须，要生成的主题个数。
    # id2word    -- 必须，LdaModel类要求我们之前的dictionary把id都映射成为字符串。
    # passes     -- 可选，模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间。
    lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics=1, id2word=dictionary, passes=1000)

    return lda_model


# 生成词云图
def cloud(texts, image_path):
    text = texts
    #     myfont = r'C:/Windows/Fonts/Arvo.TTF'
    cloudobj = wordcloud.WordCloud(font_path='msyh.ttc', width=1000, height=600, min_font_size=20, max_font_size=100,
                                   mode='RGBA', background_color=None).generate(str(text))
    plt.imshow(cloudobj)
    plt.axis('off')
    plt.show()
    plt.savefig(image_path)


# LDA文档生成模型主方法
def keywords_extraction(texts):
    words_list = get_text(texts)
    #     print('分词后的文本：')
    #     print(words_list)

    # for item in words_list:
        # print(item)

    # 获取训练后的LDA模型
    lda_model = LDA_model(words_list)

    # # 可以用 print_topic 和 print_topics 方法来查看主题
    # # 打印所有主题，每个主题显示5个词
    # topic_words = lda_model.print_topics(num_topics=1, num_words=5)
    # # print('打印所有主题，每个主题显示20个词:')
    # print(topic_words)

    # # 输出该主题的的词及其词的权重
    # words_list0 = lda_model.show_topic(0, 3)
    # print('输出该主题的的词及其词的权重:')
    # print(words_list0)
    #
    words_list1 = lda_model.show_topic(1, 5)
    # print('输出该主题的的词及其词的权重:')
    print(words_list1)


