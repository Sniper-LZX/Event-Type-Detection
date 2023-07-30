import pickle
from os import path
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import colors


# grade：图片标记
def text_by_background(text, image_path, grade):
    colors_list = ['#75bbfd']
    color_map = colors.ListedColormap(colors_list)

    backgroud_image = plt.imread(image_path)
    wc = WordCloud(
        background_color='white',  # 设置背景颜色
        mask=backgroud_image,  # 设置背景图片
        font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
        max_words=2000,  # 设置最大现实的字数
        max_font_size=50,  # 设置字体最大值
        min_font_size=20,
        colormap=color_map,
        random_state=20  # 设置有多少种随机生成状态，即有多少种配色方案
    )
    wc.generate_from_text(text)
    # print('开始加载文本')
    # # 改变字体颜色
    img_colors = ImageColorGenerator(backgroud_image)
    # 字体颜色为背景图片的颜色
    wc.recolor(color_func=img_colors)
    # 显示词云图
    plt.imshow(wc)
    # 是否显示x轴、y轴下标
    plt.axis('off')
    plt.show()
    # # 获得模块所在的路径的
    # d = path.dirname(__file__)
    # # os.path.join()：  将多个路径组合后返回
    # wc.to_file(path.join(d, "h11.jpg"))
    wc.to_file("../data/word_cloud/word_cloud"+ str(grade) + ".jpg")
    # print('生成词云成功!')